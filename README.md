# The Third Moment

The code and figures behind The Third Moment's articles. Each article explains one job on a trading
desk by building a small, working version of it, and the article's Python blocks, read in order, are
the complete program. Each folder below holds that program as one file, `article.py`, with its tests,
and running it reproduces every number, table and chart in the article.

| # | article | folder |
|---|---|---|
| 001 | How an Options Market Maker Makes (and Loses) Money | [posts/001-options-market-maker](posts/001-options-market-maker) |
| 002 | What a Volatility Trader Actually Trades | [posts/002-volatility-trader](posts/002-volatility-trader) |
| 003 | How an ETF Market Maker Keeps a Fund's Price Close to Its Value | [posts/003-etf-market-maker](posts/003-etf-market-maker) |
| 004 | Why a Good Order Book Signal Can Lose Money Once You Pay the Spread | [posts/004-hft-quant-researcher](posts/004-hft-quant-researcher) |
| 005 | What a Drawdown Stop Does to a Skilled Portfolio Manager | [posts/005-pod-portfolio-manager](posts/005-pod-portfolio-manager) |

## Run an article's code

```bash
cd posts/<folder>/code
uv run python article.py    # writes results.json and figures/
uv run pytest
```

`results.json` records a hash of the code that produced it.

## Licence

Code is MIT licensed (see LICENSE). Article text and figures are © the author, all rights reserved.
