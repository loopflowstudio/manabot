# Build the manabot paper.
#
#   make        -> paper/manabot.pdf
#   make watch  -> rebuild on save
#   make clean
#
# Requires tectonic (brew install tectonic). It fetches TeX packages on first
# run and caches them; no TeXLive install needed.

PDF := manabot.pdf

$(PDF): main.tex refs.bib
	tectonic -X compile main.tex --outdir . --keep-logs
	@mv -f main.pdf $(PDF)
	@echo "==> $(PDF)"

watch:
	tectonic -X watch

clean:
	rm -f $(PDF) main.pdf main.log main.aux main.bbl main.blg

.PHONY: watch clean
