AC_INIT([kpfft_lib], [0.1])
AC_PROG_CC

AC_ARG_ENABLE([static],
              [AS_HELP_STRING([--enable-static],
              [Use static libraries])]
              [],
              [AS_VAR_SET(enable_static,no)]
)

AC_ARG_WITH([fftw],
    [AS_HELP_STRING([--with-fftw=DIR], [look for FFTW in DIR])],
    [FFTW_LIBS="-L$withval/lib -lfftw3_threads -lfftw3" FFTW_CFLAGS="-I$withval/include" ]
)

if test x"$enable_static" != x"no"; then
	LDFLAGS="$LDFLAGS -static"
fi

if test x"$FFTW_LIBS" = x""; then
	AC_CHECK_LIB([fftw3_threads], [fftw_execute],[FFTW_LIBS="-lfftw3_threads -lfftw3" FFTW_CFLAGS="" AC_SUBST(FFTW_LIBS) AC_SUBST(FFTW_CFLAGS)],AC_MSG_ERROR([FFTW not found]), [-lfftw3])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
