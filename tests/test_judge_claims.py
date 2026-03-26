from evaluation.judge_claims import judge_claim_against_sources


def test_judge_claim_supported():
    claim = "Drug X reduces blood pressure"
    sources = [{"title": "Study on Drug X", "content": "Drug X significantly reduces blood pressure"}]
    assert judge_claim_against_sources(claim, sources) == "Supported"


def test_judge_claim_contradicted():
    claim = "Drug X does not reduce blood pressure"
    sources = [{"title": "Study on Drug X", "content": "Drug X significantly reduces blood pressure"}]
    assert judge_claim_against_sources(claim, sources) == "Contradicted"


def test_judge_claim_nei():
    claim = "Drug X cures diabetes"
    sources = [{"title": "Study on Drug X", "content": "Drug X reduces blood pressure"}]
    assert judge_claim_against_sources(claim, sources) == "NEI"
