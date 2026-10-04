# site/

`index.html` is the standalone HTML build of [`../RESULTS.md`](../RESULTS.md) — the
hirer-facing write-up — and it is what the `gh-pages` branch serves at
<https://lovemanna.github.io/deltaforge/>.

It is one self-contained file: all CSS is inline, the only external request is the Google
Fonts stylesheet, and there is no build step and no JavaScript beyond a ~25-line tooltip
handler at the bottom.

**To change the page**, edit `index.html` here, commit it on the working branch, then
publish it:

```sh
git worktree add --detach /tmp/df-pages
git -C /tmp/df-pages switch gh-pages
cp site/index.html /tmp/df-pages/index.html
git -C /tmp/df-pages commit -am "Update the page"
git -C /tmp/df-pages push
git worktree remove /tmp/df-pages
```

`gh-pages` is an orphan branch holding exactly `index.html` and `.nojekyll` — no history
from `main`, and nothing on it to merge back. `.nojekyll` is load-bearing: without it
GitHub runs Jekyll over the branch.

**Keep the two in step.** `RESULTS.md` and `index.html` carry the same numbers in the same
order; if a rental changes a headline, change both or the public page starts contradicting
the repository.
