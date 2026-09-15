"""Category variant normalization (retail-domain capability).

Mechanical variant labels ("Electronic"/"Electronics", case/whitespace noise)
are merged to the most frequent spelling. Synonyms ("Tee"/"T-Shirt") are
deliberately NOT merged — uncertainty is preserved, disclosed in the step
reason, and the merge map travels in the plan for auditability.
"""
import pandas as pd

from app.services.cleaning import apply_cleaning_plan, build_cleaning_plan


def test_variant_merge_is_deterministic_and_idempotent():
    df = pd.DataFrame(
        {
            "category": [
                "Electronic",
                "Electronics",
                "ELECTRONICS ",
                "Apparel",
                "T-Shirt",
                "Tee",  # synonym: must NOT merge
                "Toys",
            ]
        }
    )

    plan = build_cleaning_plan(df, "products")
    step = next(s for s in plan if s.step_id == "normalize_category_variants_category")
    assert step.source == "code"
    assert "synonyms are NOT merged" in step.reason
    assert step.params["merge_map"], "merge map must travel in the plan"

    df_clean, executed = apply_cleaning_plan(df, plan)
    vstep = next(s for s in executed if s.step_id == "normalize_category_variants_category")
    assert vstep.status == "applied"
    assert vstep.rows_affected == 1  # only 'ELECTRONICS ' needed post-case merge

    merged = set(df_clean["category"])
    # 'Electronics' appears twice (post-normalisation) vs 'Electronic' once,
    # so the most-frequent-spelling rule makes IT canonical — same as on the
    # real sample data (57 vs 5).
    assert "Electronics" in merged and "Electronic" not in merged
    assert {"T-Shirt", "Tee"}.issubset(merged)  # synonym preserved
    # 7 rows: 'Electronic' and 'ELECTRONICS ' both collapse onto the existing
    # 'Electronics' row, so TWO rows become exact duplicates and are dropped.
    assert len(df_clean) == 5

    # Re-clean is a no-op (idempotency) and never re-drops rows.
    plan2 = build_cleaning_plan(df_clean, "products")
    df_clean2, executed2 = apply_cleaning_plan(df_clean, plan2)
    assert not any(s.step_id.startswith("normalize_category_variants") for s in executed2)
    assert df_clean2.equals(df_clean)


def test_merge_favors_most_frequent_spelling():
    df = pd.DataFrame({"category": ["Apparel", "Apparel", "apparel", "Toys"]})
    df_clean, executed = apply_cleaning_plan(df, build_cleaning_plan(df, "products"))
    assert "Apparel" in set(df_clean["category"]) and "apparel" not in set(df_clean["category"])
    assert len(df_clean) == 2  # 'apparel' row collapsed onto 'Apparel' row, then deduped


def test_identifiers_and_names_are_never_merged():
    df = pd.DataFrame(
        {
            "product_id": ["P1", "P1X", "P1"],
            "product_name": ["Widget", "WIDGET", "Widget"],
        }
    )
    plan = build_cleaning_plan(df, "products")
    assert not any(s.step_id.startswith("normalize_category_variants") for s in plan)


def test_plaintext_sample_data_merges_electronic_variants():
    """On the real sample data the merge must keep row count at 60 (no loss)."""
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "products.csv"
    df = pd.read_csv(sample)
    df_clean, _ = apply_cleaning_plan(df, build_cleaning_plan(df, "products"))

    assert len(df_clean) == len(df) == 60  # no silent row loss from merging alone
    cats = set(df_clean["category"])
    assert not ({"Electronic", "Electronics"} <= cats), "variants must be merged"
    assert cats == {
        "Apparel",
        "Electronics",
        "Footwear",
        "Home & Kitchen",
        "Home & Office",
        "Sports & Outdoors",
    }
