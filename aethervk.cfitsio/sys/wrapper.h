/* bindgen entry point: the library is always linked statically, so the export macro expands to nothing. */
#define CFITSIO_STATIC_DEFINE 1
#include <fitsio.h>
/* the internal header: the in-memory gzip codec (zcompress.c) is declared here only */
#include <fitsio2.h>
/* libbz2 (the cfitsio `bzip2` feature): the one-shot buffer codec for .fits.bz2 */
#include <bzlib.h>
