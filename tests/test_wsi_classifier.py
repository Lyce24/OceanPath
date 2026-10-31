"""
Tests for loading pretrained aggregator weights into the classifier
(``oceanpath.models.wsi_classifier.WSIClassifier``), the fine-tuning path that
``training.aggregator_weights_path`` drives, and for re-initialisation.
Forward shapes and freezing are covered in tests/test_models.py.

Covers:
  * load_aggregator_weights round trips: the bare state dict scripts/pretrain.py
    writes as aggregator_weights.pt, {"state_dict"} / {"model_state_dict"}
    wrappers, a key prefix stripped from a full Lightning checkpoint
  * strict loading rejects mismatched keys; non-strict loading keeps the rest
  * MILTrainModule(aggregator_weights_path=...) starting from those weights
  * initialize_weights: redraws aggregator and head, but leaves a frozen
    (pretrained) aggregator alone
"""

import pytest
import torch

from oceanpath.models import build_classifier
from oceanpath.modules.train_module import MILTrainModule

IN_DIM = 16
TINY_ABMIL = {"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}


def make_classifier(seed: int, **kwargs):
    torch.manual_seed(seed)
    return build_classifier(
        "abmil", in_dim=IN_DIM, num_classes=2, model_cfg=dict(TINY_ABMIL), **kwargs
    )


def clone_state(module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def assert_same_state(a: dict, b: dict):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.fixture
def pretrained(tmp_path):
    """A 'pretrained' aggregator and the bare state dict file scripts/pretrain.py writes."""
    source = make_classifier(seed=1)
    path = tmp_path / "aggregator_weights.pt"
    torch.save(source.aggregator.state_dict(), path)
    return source, path


class TestLoadAggregatorWeights:
    def test_bare_state_dict_round_trip(self, pretrained):
        source, path = pretrained
        target = make_classifier(seed=2)
        head_before = clone_state(target.head)
        target.load_aggregator_weights(str(path))
        assert_same_state(clone_state(target.aggregator), clone_state(source.aggregator))
        assert_same_state(clone_state(target.head), head_before)  # the head stays as it was
        x = torch.randn(1, 9, IN_DIM)
        with torch.no_grad():
            torch.testing.assert_close(
                target.aggregator(x).slide_embedding, source.aggregator(x).slide_embedding
            )

    @pytest.mark.parametrize("wrapper", ["state_dict", "model_state_dict"])
    def test_wrapped_checkpoints(self, pretrained, tmp_path, wrapper):
        source, _ = pretrained
        path = tmp_path / "wrapped.pt"
        torch.save({wrapper: source.aggregator.state_dict(), "epoch": 3}, path)
        target = make_classifier(seed=2)
        target.load_aggregator_weights(str(path))
        assert_same_state(clone_state(target.aggregator), clone_state(source.aggregator))

    def test_prefix_is_stripped_from_a_lightning_checkpoint(self, tmp_path):
        torch.manual_seed(1)
        trained = MILTrainModule(
            arch="abmil", in_dim=IN_DIM, num_classes=2, model_cfg=dict(TINY_ABMIL)
        )
        path = tmp_path / "fold.ckpt"
        torch.save(
            {"state_dict": trained.state_dict()}, path
        )  # keys: model.aggregator.*, model.head.*
        target = make_classifier(seed=2)
        head_before = clone_state(target.head)
        target.load_aggregator_weights(str(path), prefix="model.aggregator.")
        assert_same_state(clone_state(target.aggregator), clone_state(trained.model.aggregator))
        assert_same_state(clone_state(target.head), head_before)

    def test_strict_loading_rejects_a_mismatch_and_lenient_loading_keeps_the_rest(
        self, pretrained, tmp_path
    ):
        source, _ = pretrained
        partial = dict(source.aggregator.state_dict())
        dropped = sorted(partial)[0]
        del partial[dropped]
        path = tmp_path / "partial.pt"
        torch.save(partial, path)
        with pytest.raises(RuntimeError, match="Missing key"):
            make_classifier(seed=2).load_aggregator_weights(str(path), strict=True)
        target = make_classifier(seed=2)
        own = clone_state(target.aggregator)
        target.load_aggregator_weights(str(path))
        loaded = clone_state(target.aggregator)
        assert torch.equal(loaded[dropped], own[dropped])
        for key, value in partial.items():
            assert torch.equal(loaded[key], value), key

    def test_training_module_starts_from_the_pretrained_aggregator(self, pretrained):
        source, path = pretrained
        torch.manual_seed(2)
        module = MILTrainModule(
            arch="abmil",
            in_dim=IN_DIM,
            num_classes=2,
            model_cfg=dict(TINY_ABMIL),
            aggregator_weights_path=str(path),
        )
        assert_same_state(clone_state(module.model.aggregator), clone_state(source.aggregator))


class TestInitializeWeights:
    def _zero(self, module):
        with torch.no_grad():
            for p in module.parameters():
                p.zero_()

    def test_redraws_aggregator_and_head(self):
        model = make_classifier(seed=0)
        self._zero(model)
        model.initialize_weights()
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                assert param.abs().sum() > 0, name

    def test_leaves_a_frozen_aggregator_alone(self, pretrained):
        source, path = pretrained
        model = make_classifier(seed=2, freeze_aggregator=True)
        model.load_aggregator_weights(str(path))
        self._zero(model.head)
        model.initialize_weights()
        assert_same_state(clone_state(model.aggregator), clone_state(source.aggregator))
        assert model.head.weight.abs().sum() > 0
