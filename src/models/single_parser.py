import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from supar import CRFDependencyParser, CRFConstituencyParser
from supar.utils import Dataset
from supar.utils.logging import get_logger, init_logger
from supar.utils.fn import set_rng_state
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.parallel import is_dist
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import Batch, Tree
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.fn import ispunct
from supar.utils.metric import SpanMetric

from models.sigle import MaxMarginConstModel, MaxMarginDepModel
from utils.metric import DELETE_SPMRL, EQUAL_SPMRL, DELETE_HEBREW, EQUAL_HEBREW


logger = get_logger(__name__)


class OptCRFDependencyParser(CRFDependencyParser):
    NAME = 'mm-dependency'
    MODEL = MaxMarginDepModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
            punct=False, mbr=True, tree=False, proj=False, partial=False, verbose=True, clip=5.0, epochs=5000, patience=100, 
            **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
        eval_batch_size = args.get('eval_batch_size', batch_size)
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        train = Dataset(self.transform, args.train, **args).build(batch_size, buckets, True, is_dist(), workers)
        
        global_proj = args.proj
        args.proj = False
        
        dev = Dataset(self.transform, args.dev, **args).build(eval_batch_size, buckets, False, is_dist(), workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, **args).build(eval_batch_size, buckets, False, is_dist(), workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")
        
        args.proj = global_proj
        
        loader, sampler = train.loader, train.loader.batch_sampler

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        elif args.encoder == 'transformer':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = InverseSquareRootLR(self.optimizer, args.warmup_steps)
        else:
            # we found that Huggingface's AdamW is more robust and empirically better than the native implementation
            from transformers import AdamW
            steps = len(train.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr,
                (args.mu, args.nu),
                args.eps,
                args.weight_decay
            )
            self.scheduler = LinearLR(self.optimizer, int(steps*args.warmup), steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)

        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(loader)
        self.total_steps = self.n_batches * epochs // args.update_steps
        self.best_metric, self.elapsed = Metric(), timedelta()
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                sampler.set_epoch(self.epoch)
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should zero `step` as the number of batches in different processes is not necessarily equal
                self.step = 0
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device, enabled=self.args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(), self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device, enabled=self.args.amp):
                metric = self.reduce(sum([self.eval_step(i) for i in progress_bar(dev.loader)], Metric()))
                logger.info(f"{'dev:':5} {metric}")
                if args.test:
                    test_metric = sum([self.eval_step(i) for i in progress_bar(test.loader)], Metric())
                    logger.info(f"{'test:':5} {self.reduce(test_metric)}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if is_dist():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum([best.eval_step(i) for i in progress_bar(test.loader)], Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        words, _, *feats, arcs, rels = batch
        mask = batch.mask
        # ignore the first token of each sentence
        mask[:, 0] = 0
        s_arc, s_rel = self.model(words, feats)
        loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.mbr, self.args.partial)
        arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
        if self.args.partial:
            mask &= arcs.ge(0)
        # ignore all punctuation if not specified
        if not self.args.punct:
            mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in batch.sentences for w in s.words]))
        return AttachmentMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask)


class OptCRFConstituencyParser(CRFConstituencyParser):
    NAME = 'mm-constituency'
    MODEL = MaxMarginConstModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, _, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_label = self.model(words, feats)
        loss = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanMetric:
        words, *feats, trees, charts = batch
        mask = batch.mask[:, 1:]
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        s_span, s_label = self.model(words, feats)
        loss = self.model.loss(s_span, s_label, charts, mask, self.args.mbr)
        chart_preds = self.model.decode(s_span, s_label, mask)
        preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                 for tree, chart in zip(trees, chart_preds)]
        if self.args.spmrl:
            # follow https://github.com/nikitakit/self-attentive-parser/tree/master
            # donot use hebrew rules
            delete = DELETE_SPMRL  # if self.args.lan != 'Hebrew' else DELETE_HEBREW
            equal = EQUAL_SPMRL  # if self.args.lan != 'Hebrew' else EQUAL_HEBREW
            return SpanMetric(loss,
                              [Tree.factorize(tree, delete, equal) for tree in preds],
                              [Tree.factorize(tree, delete, equal) for tree in trees])
        return SpanMetric(loss,
                          [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                          [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
