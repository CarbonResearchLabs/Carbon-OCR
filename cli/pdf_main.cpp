// carbon-pdf: a helper process of carbon::PdfPool (carbon/pdf.h), which renders
// PDF pages for it, one request at a time on its standard input and output.  Not
// for running by hand.

#include "carbon/pdf.h"

int main() { return carbon::run_pdf_helper(); }
