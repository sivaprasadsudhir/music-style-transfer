all: proposal

proposal:
	pdflatex proposal.tex
	bibtex proposal
	pdflatex proposal.tex
	pdflatex proposal.tex

clean:
	rm -rf *.out *.log *.pdf *.blg *.bbl *.aux