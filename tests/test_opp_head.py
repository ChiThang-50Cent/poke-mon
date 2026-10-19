"""W6-A opponent head: learner-view support and the loss only count labelled rows."""

import torch

from pokeai.training.opp_head import move_class_mask, opponent_loss, support, val_half

VOCAB = ["move:<unk>", "switch:<unk>", "move:earthquake", "switch:rotom", "switch:garchomp",
         "switch:zapdos"]


def test_support_is_moves_plus_preview_minus_active():
    always = move_class_mask(VOCAB)
    preview = torch.tensor([[3, 4, -1, -1, -1, -1]])
    mask = support(preview, torch.tensor([4]), always)
    # zapdos is not in the preview, garchomp is active, padding adds nothing
    assert mask.tolist() == [[True, True, True, True, False, False]]
    assert support(preview, torch.tensor([-1]), always)[0, 4]


def test_loss_ignores_unlabelled_and_padding_rows():
    always = move_class_mask(VOCAB)
    logits = torch.zeros(1, 3, len(VOCAB) + 1)
    batch = {
        "opp_class": torch.tensor([[2, -1, 3]]), "opp_tera": torch.tensor([[1, -1, -1]]),
        "opp_preview": torch.tensor([[[3, 4, -1, -1, -1, -1]] * 3]),
        "opp_active": torch.tensor([[4, 4, 4]]),
        "real": torch.tensor([[True, True, False]]),
    }
    ce, bce, n, n_tera = opponent_loss(logits, batch, always)
    assert (n, n_tera) == (1, 1)
    assert torch.isclose(ce, torch.log(torch.tensor(4.0)))  # uniform over 4 supported
    assert torch.isclose(bce, torch.log(torch.tensor(2.0)))


def test_val_halves_are_battle_level_and_stable():
    assert val_half("gen9ou-1") == val_half("gen9ou-1")
    assert {val_half(f"gen9ou-{i}") for i in range(50)} == {"A", "B"}
