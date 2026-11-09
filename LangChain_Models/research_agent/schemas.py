from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """A structured summary of ONE research paper, written only from its text."""

    title: str = Field(description="The paper's title.")
    problem: str = Field(description="What problem the paper tries to solve, in 1-3 sentences.")
    method: str = Field(description="The proposed approach / model / technique, in 2-5 sentences.")
    datasets: list[str] = Field(
        description="Datasets or benchmarks used. Use ['Insufficient information available'] if none are mentioned."
    )
    key_results: list[str] = Field(description="The main quantitative or qualitative results, 2-5 bullet points.")
    limitations: list[str] = Field(
        description="Limitations stated or clearly implied by the paper. Use ['Insufficient information available'] if none."
    )
    key_equations: list[str] = Field(
        description="Up to 3 important equations or formulas written in plain text/LaTeX, each with a short meaning. Empty list if none."
    )
    one_line_takeaway: str = Field(description="One sentence a busy reader should remember.")


class ComparisonRow(BaseModel):
    """One paper's values for every comparison dimension."""

    arxiv_id: str
    title: str
    cells: list[str] = Field(
        description="One short value per dimension, in the SAME order as the dimensions list."
    )


class PaperComparison(BaseModel):
    """A side-by-side comparison of several papers."""

    dimensions: list[str] = Field(
        description="Comparison columns, e.g. ['Method', 'Datasets', 'Key result', 'Limitations']."
    )
    rows: list[ComparisonRow]
    overall_insight: str = Field(
        description="2-4 sentences: how the approaches differ, which looks strongest and what gap remains."
    )
