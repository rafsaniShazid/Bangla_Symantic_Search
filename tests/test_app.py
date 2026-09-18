from pathlib import Path

import pandas as pd

import app


def test_cli_tfidf_search_with_fixture(tmp_path: Path, capsys) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {"title": "বাংলাদেশ বাজেট", "content": "বাজেট ঘোষণা", "category": "দেশ"},
            {"title": "ক্রিকেট", "content": "ম্যাচ জয়", "category": "খেলা"},
        ]
    ).to_csv(path, index=False)

    exit_code = app.run_cli("বাংলাদেশ বাজেট", "tfidf", 1, path)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Method: TF-IDF" in output
    assert "বাংলাদেশ বাজেট" in output


def test_cli_rejects_unknown_method() -> None:
    try:
        app.build_searcher("unknown")
    except ValueError as error:
        assert "Unknown method" in str(error)
    else:
        raise AssertionError("Expected an unknown-method error")
