#pragma once

#define SCEIL(x,n) ((x + (1 << n) - 1) >> n)
#define XCEIL(x,n) ( SCEIL(x,n) << n)