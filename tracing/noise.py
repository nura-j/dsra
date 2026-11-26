"""Noise for corrupting the definition embeddings.

The default rule is ROME's spherical Gaussian scaled to the embedding spread:
``"s3"`` adds N(0, (3 sigma_e)^2), where sigma_e is the standard deviation of
the embeddings of the traced definitions. The multivariate (``"m"``) and
Student-t (``"t<df>"``) rules sample from the embedding covariance, estimated
on WikiText-103.
"""

import numpy
import torch
from datasets import load_dataset

from legacy.rome import nethook
from legacy.rome.runningstats import Covariance, tally
from legacy.rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

from .model import layername, make_inputs


def collect_embedding_std(mt, subjects):
    """Standard deviation of the input embeddings over the given texts."""
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s], device=mt.device)
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    return alldata.std().item()


def get_embedding_cov(mt):
    """Mean and covariance of the input embeddings over WikiText-103."""
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except AttributeError:
            maxlen = 100  # GPT-NeoX configs have no n_positions.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=None,
        sample_size=1000,
        batch_size=5,
        collate_fn=length_collation(100),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, mt.device)
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    """Linear map turning N(0, I) samples into N(mean, cov) samples."""
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    """Noise transform matched to the embedding mean and covariance."""
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    """Student-t noise with the embedding covariance and matched variance.

    Samples sqrt((degree - 2) / u) * g with u ~ chi2(degree) and g from the
    matched Gaussian, so the variance equals the embedding covariance.
    """
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        return factor * gauss

    return normal_to_student


def resolve_noise(mt, rule, subjects):
    """Turn a noise rule into ``(noise, uniform_noise)`` for the tracer.

    ``noise`` is a float scale or a callable transform; ``uniform_noise`` says
    whether the base samples are uniform rather than Gaussian.
    """
    if not isinstance(rule, str):
        return float(rule), False
    if rule.startswith("s"):
        factor = float(rule[1:]) if len(rule) > 1 else 1.0
        noise = factor * collect_embedding_std(mt, subjects)
        print(f"Using noise_level {noise} to match model times {factor}")
        return noise, False
    if rule == "m":
        print("Using multivariate gaussian to match model noise")
        return collect_embedding_gaussian(mt), False
    if rule.startswith("t"):
        return collect_embedding_tdist(mt, float(rule[1:])), False
    if rule.startswith("u"):
        return float(rule[1:]), True
    raise ValueError(f"unknown noise rule {rule!r}")
