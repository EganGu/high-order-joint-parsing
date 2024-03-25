from __future__ import annotations

import os
from datetime import datetime, timedelta
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import shutil
import tempfile

from supar.parser import Parser
from supar.utils import Config, Embedding
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.transform import CoNLL, Tree
from supar.utils.fn import set_rng_state, ispunct
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import gather, is_dist, is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.optim import InverseSquareRootLR, LinearLR

from .joint import JointModel
from utils.data import JointDataset
from utils.transform import JointTrans, JointBatch
from utils.metric import JointMetric, DELETE_SPMRL, EQUAL_SPMRL, DELETE_HEBREW, EQUAL_HEBREW


logger = get_logger(__name__)


class JointParser(Parser):
    r"""The implementation of dep-con joint parser. """
    NAME = 'joint'
    MODEL = JointModel

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

        self.TAG = self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART

    @classmethod
    def build(
        cls,
        path: str,
        min_freq: int = 2,
        fix_len: int = 20,
        verbose: int = True,
        **kwargs
    ) -> JointParser:
        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(
                parser.transform.FORM[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True)
        TAG, CHAR, ELMO, BERT = None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk,
                                bos=t.bos, eos=t.eos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK,
                         bos=BOS, eos=EOS, lower=True)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK,
                                    bos=BOS, eos=EOS, fix_len=args.fix_len)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos,
                                    eos=t.eos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        if 'tag' in args.feat:
            TAG = Field('tags', bos=BOS, eos=EOS)
        # dep transform: conll
        TEXT = RawField('texts')
        ARC = Field('arcs', bos=BOS, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=BOS)
        # con transform: tree
        TREE = RawField('trees')
        CHART = ChartField('charts')
        HEADED = ChartField('heads', use_vocab=False)
        HOOK = ChartField('hooks', use_vocab=False)

        transform = JointTrans(FORM=(WORD, TEXT, CHAR, ELMO, BERT), CPOS=TAG, HEAD=ARC, DEPREL=REL,
                               TREE=TREE, CHART=CHART, HEADED=HEADED, HOOK=HOOK, binarize_way=args.binarize_way)
        train = JointDataset(transform, args.train_dep, args.train_con, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(
                args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if CHAR is not None:
                CHAR.build(train)
        if TAG is not None:
            TAG.build(train)
        REL.build(train)
        CHART.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_labels': len(CHART.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(
            **args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser

    def train(
        self,
        buckets: int = 32,
        workers: int = 0,
        batch_size: int = 5000,
        update_steps: int = 1,
        clip: int = 5.0,
        epochs: int = 5000,
        patience: int = 100,
        amp: bool = False,
        cache: bool = False,
        mbr: bool = True,
        delete: set = {'TOP', 'S1', '-NONE-', ',',
                       ':', '``', "''", '.', '?', '!', ''},
        equal: set = {'ADVP': 'PRT'},
        verbose: bool = True,
        punct: bool = False,
        both_decode: bool = True,
        proj: bool = False,
        **kwargs
    ) -> None:
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        train = JointDataset(self.transform, args.train_dep, args.train_con, **
                             args).build(batch_size, buckets, True, dist.is_initialized(), workers)
        global_proj = args.proj
        args.proj = False
        dev = JointDataset(self.transform, args.dev_dep, args.dev_con, **args).build(
            batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test_dep:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = JointDataset(self.transform, args.test_dep, args.test_con, **args).build(
                batch_size, buckets, False, dist.is_initialized(), workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")
        args.proj = global_proj

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(
            ), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(
                self.optimizer, args.decay**(1/args.decay_steps))
        elif args.encoder == 'transformer':
            self.optimizer = Adam(self.model.parameters(
            ), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = InverseSquareRootLR(
                self.optimizer, args.warmup_steps)
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
            self.scheduler = LinearLR(
                self.optimizer, int(steps*args.warmup), steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get(
                                 'find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(
                    dist.group.WORLD, fp16_compress_hook)

        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(
            train.loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(
                    self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(
                    self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(
                    self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                train.loader.batch_sampler.epoch = self.epoch
            except AttributeError:
                logger.warning(
                    "No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(train.loader), Metric()

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
                        self.clip_grad_norm_(
                            self.model.parameters(), self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(
                        f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device, enabled=self.args.amp):
                if args.binarize_way == 'head':
                    metric = self.reduce(
                        sum([self.eval_step(i) for i in progress_bar(dev.loader)], Metric()))
                elif args.binarize_way == 'lr':
                    metric = self.reduce(sum(
                        [self.eval_step(i, decoding='idp') for i in progress_bar(dev.loader)], Metric()))
                else:
                    raise ValueError
                logger.info(f"{'dev:':5} {metric}")
                if args.test_dep:
                    test_metric_satta = sum(
                        [self.eval_step(i) for i in progress_bar(test.loader)], Metric())
                    logger.info(
                        f"{'test[satta]:':15} \n {self.reduce(test_metric_satta)}")
                    if args.both_decode:
                        test_metric_idp = sum(
                            [self.eval_step(i, decoding='idp') for i in progress_bar(test.loader)], Metric())
                        logger.info(
                            f"{'test[idp]:':15} \n {self.reduce(test_metric_idp)}")

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
        if dist.is_initialized():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test_dep:
            best.model.eval()
            with best.join():
                test_metric_satta = sum(
                    [best.eval_step(i) for i in progress_bar(test.loader)], Metric())
                logger.info(
                    f"{'test[satta]:':15} \n {self.reduce(test_metric_satta)}")
                if args.both_decode:
                    test_metric_idp = sum(
                        [best.eval_step(i, decoding='idp') for i in progress_bar(test.loader)], Metric())
                    logger.info(
                        f"{'test[idp]:':15} \n {self.reduce(test_metric_idp)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def train_step(self, batch: JointBatch) -> torch.Tensor:
        words, _, *feats, arcs, rels, trees, charts, heads, hooks = batch
        mask_dep, mask_con = batch.mask_dep, batch.mask_con
        s_lex, s_tag, s_vi2o = self.model(words, feats)
        gold = (arcs, charts, rels, heads, hooks) if self.model.args.use_head else (
            arcs, charts, rels)
        loss, _ = self.model.loss(
            s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: JointBatch, decoding: str = 'satta') -> JointMetric:
        words, texts, *feats, arcs, rels, trees, charts, heads, hooks = batch
        # for te, tr in zip(texts, trees):
        #     if tuple([x.replace('\\', '') for x in tr.leaves()]) != te:
        #         breakpoint()
        mask_dep, mask_con = batch.mask_dep, batch.mask_con
        s_lex, s_tag, s_vi2o = self.model(words, feats)
        gold = (arcs, charts, rels, heads, hooks) if self.model.args.use_head else (
            arcs, charts, rels)
        # if self.model.args.loss_type == 'mm':
        loss, _ = self.model.loss(
            s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con)
        # else:
        #     loss, s_lex = self.model.loss(s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con)
        arc_preds, rel_preds, chart_preds = self.model.decode(s_lex, s_tag, mask_dep, mask_con, decoding=decoding)
        if self.args.partial:
            mask_dep &= arcs.ge(0)
        # ignore all punctuation if not specified
        if not self.args.punct:
            mask_dep.masked_scatter_(mask_dep, ~mask_dep.new_tensor(
                [ispunct(w) for s in batch.sentences for w in s.words]))
        preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                 for tree, chart in zip(trees, chart_preds)]
        # loss = loss.sum()
        if hasattr(self.args, 'spmrl') and self.args.spmrl:
            # follow https://github.com/nikitakit/self-attentive-parser/tree/master
            # donot use hebrew rules
            delete = DELETE_SPMRL  # if self.args.lan != 'Hebrew' else DELETE_HEBREW
            equal = EQUAL_SPMRL  # if self.args.lan != 'Hebrew' else EQUAL_HEBREW
            return JointMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask_dep,
                               [Tree.factorize(tree, delete, equal) for tree in preds],
                               [Tree.factorize(tree, delete, equal) for tree in trees])
        return JointMetric(loss, (arc_preds, rel_preds), (arcs, rels), mask_dep,
                           [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                           [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])

    @torch.no_grad()
    def pred_step(self, batch: JointBatch, decoding: str = 'satta',
                  non_label=True, mark='none', ana_mode=False) -> JointMetric:
        words, texts, *feats, trees = batch
        mask_dep, mask_con = batch.mask_dep, batch.mask_con
        lens = (batch.lens - 2).tolist()
        s_lex, s_tag, s_vi2o = self.model(words, feats)
        arc_preds, rel_preds, chart_preds = self.model.decode(
            s_lex, s_tag, mask_dep, mask_con, decoding=decoding)
        # if not self.args.punct:
        #     mask_dep.masked_scatter_(mask_dep, ~mask_detrep.new_tensor(
        #         [ispunct(w) for s in batch.sentences for w in s.words]))
        for te, tr in zip(texts, trees):
            try:
                if tuple([x.replace('\\', '') for x in tr.leaves()]) != te:
                    breakpoint()
            except AttributeError:
                breakpoint()

        preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
                 for tree, chart in zip(trees, chart_preds)]
        if ana_mode:
            preds_binarized = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart], mark=mark)
                               for tree, chart in zip(trees, chart_preds)]
        batch.arcs = [i.tolist() for i in arc_preds[mask_dep].split(lens)]
        batch.rels = [self.REL.vocab[i.tolist()]
                      for i in rel_preds[mask_dep].split(lens)]
        batch.trees = preds

        if ana_mode:
            return (arc_preds, rel_preds), (preds, preds_binarized), mask_dep
        return batch

    def predict(
        self,
        data_dep: str,
        data_con: str,
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        cache: bool = False,
        **kwargs
    ):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        data = JointDataset(self.transform, **args)
        data.build(batch_size, buckets, False, is_dist(), workers)
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.model.eval()
        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch, decoding=self.args.decoding)
                if args.cache:
                    for s in batch.sentences:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if is_dist():
                dist.barrier()
            if args.cache:
                tdirs = gather(t) if is_dist() else (t,)
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if args.cache:
                        sentences = (os.path.join(i, s)
                                     for i in tdirs for s in os.listdir(i))
                        for i in progress_bar(sorted(sentences, key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if is_dist():
                dist.barrier()
        logger.info(
            f"{elapsed}s elapsed, {len(data) / elapsed.total_seconds():.2f} Sents/s")

        if not cache:
            return data

    def evaluate(
        self,
        data_dep: str,
        data_con: str,
        decoding: str = 'satta',
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        delete={'TOP', 'S1', '-NONE-', ',',
                ':', '``', "''", '.', '?', '!', ''},
        equal={'ADVP': 'PRT'},
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        data = JointDataset(self.transform, **args)
        data.build(batch_size, buckets, False, is_dist(), workers)
        logger.info(f"\n{data}")

        logger.info("Evaluating the data")
        start = datetime.now()
        self.model.eval()
        with self.join():
            metric = self.reduce(
                sum([self.eval_step(i, decoding=decoding) for i in progress_bar(data.loader)], Metric()))
        elapsed = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(
            f"{elapsed}s elapsed, {len(data)/elapsed.total_seconds():.2f} Sents/s")

        return metric
