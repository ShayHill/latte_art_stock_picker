#!python
"""Use the mysterious power of divination to select stocks from an image of latte foam.

Deterministically convert an input image into a 49-bit number based on shapes, then
interpolate between `[0, math.comb(number_of_stocks, N_PICKS))` to select a
combination of stocks.

:author: Shay Hill
:created: 2023-07-31
"""


import argparse
import itertools as it
import math
import os

import numpy as np
from PIL import Image

# the number of stocks we will pick.
N_PICKS = 4

# As an intermediate step, the image will be converted into an SR_BITS**2 bit number.
# Don't change this!
SR_BITS = 7

# S&P 500 stocks per https://en.wikipedia.org/wiki/S%26P_500
SP_500 = """MMM AOS ABT ABBV ACN ATVI ADM ADBE ADP AAP AES AFL A APD AKAM ALK ALB ARE
    ALGN ALLE LNT ALL GOOGL GOOG MO AMZN AMCR AMD AEE AAL AEP AXP AIG AMT AWK AMP ABC
    AME AMGN APH ADI ANSS AON APA AAPL AMAT APTV ACGL ANET AJG AIZ T ATO ADSK AZO AVB
    AVY AXON BKR BALL BAC BBWI BAX BDX WRB BRK.B BBY BIO TECH BIIB BLK BK BA BKNG BWA
    BXP BSX BMY AVGO BR BRO BF.B BG CHRW CDNS CZR CPT CPB COF CAH KMX CCL CARR CTLT
    CAT CBOE CBRE CDW CE CNC CNP CDAY CF CRL SCHW CHTR CVX CMG CB CHD CI CINF CTAS
    CSCO C CFG CLX CME CMS KO CTSH CL CMCSA CMA CAG COP ED STZ CEG COO CPRT GLW CTVA
    CSGP COST CTRA CCI CSX CMI CVS DHI DHR DRI DVA DE DAL XRAY DVN DXCM FANG DLR DFS
    DIS DG DLTR D DPZ DOV DOW DTE DUK DD DXC EMN ETN EBAY ECL EIX EW EA ELV LLY EMR
    ENPH ETR EOG EPAM EQT EFX EQIX EQR ESS EL ETSY EG EVRG ES EXC EXPE EXPD EXR XOM
    FFIV FDS FICO FAST FRT FDX FITB FSLR FE FIS FI FLT FMC F FTNT FTV FOXA FOX BEN
    FCX GRMN IT GEHC GEN GNRC GD GE GIS GM GPC GILD GL GPN GS HAL HIG HAS HCA PEAK
    HSIC HSY HES HPE HLT HOLX HD HON HRL HST HWM HPQ HUM HBAN HII IBM IEX IDXX ITW
    ILMN INCY IR PODD INTC ICE IFF IP IPG INTU ISRG IVZ INVH IQV IRM JBHT JKHY J JNJ
    JCI JPM JNPR K KDP KEY KEYS KMB KIM KMI KLAC KHC KR LHX LH LRCX LW LVS LDOS LEN
    LNC LIN LYV LKQ LMT L LOW LYB MTB MRO MPC MKTX MAR MMC MLM MAS MA MTCH MKC MCD
    MCK MDT MRK META MET MTD MGM MCHP MU MSFT MAA MRNA MHK MOH TAP MDLZ MPWR MNST MCO
    MS MOS MSI MSCI NDAQ NTAP NFLX NWL NEM NWSA NWS NEE NKE NI NDSN NSC NTRS NOC NCLH
    NRG NUE NVDA NVR NXPI ORLY OXY ODFL OMC ON OKE ORCL OGN OTIS PCAR PKG PANW PARA
    PH PAYX PAYC PYPL PNR PEP PFE PCG PM PSX PNW PXD PNC POOL PPG PPL PFG PG PGR PLD
    PRU PEG PTC PSA PHM QRVO PWR QCOM DGX RL RJF RTX O REG REGN RF RSG RMD RVTY RHI
    ROK ROL ROP ROST RCL SPGI CRM SBAC SLB STX SEE SRE NOW SHW SPG SWKS SJM SNA SEDG
    SO LUV SWK SBUX STT STLD STE SYK SYF SNPS SYY TMUS TROW TTWO TPR TRGP TGT TEL TDY
    TFX TER TSLA TXN TXT TMO TJX TSCO TT TDG TRV TRMB TFC TYL TSN USB UDR ULTA UNP
    UAL UPS URI UNH UHS VLO VTR VRSN VRSK VZ VRTX VFC VTRS VICI V VMC WAB WBA WMT WBD
    WM WAT WEC WFC WELL WST WDC WRK WY WHR WMB WTW GWW WYNN XEL XYL YUM ZBRA ZBH ZION
    ZTS""".split()


# ensure system floating point precision is at least 49 bits.
_msg = (
    "SR_BITS is too large for system precision. Do not change SR_BITS. "
    + "Run this on another system to ensure results are deterministic given an image."
)
assert (2**SR_BITS - 1) / 2**SR_BITS != 1, _msg


# =============================================================================
#
# Select a combination of stocks
#
# =============================================================================


def _get_ith_comb(n: int, k: int, idx: int) -> list[int]:
    """Get the idx'th combination of k elements from n elements.

    :param n: The number of elements to choose from.
    :param k: The number of elements to choose.
    :param idx: The index of the combination to get.
    :return: The idx'th ordered combination of k elements from n elements.

    Equivalent to `list(list(it.combinations(n, k))[idx])` without having to generate
    the entire list.
    """
    result: list[int] = []
    n_p = n
    k_p = k

    def get_below(n_: int) -> int:
        """Get comb(n_, kp) where kp is in outer scope."""
        return math.comb(n_, k_p)

    above = (math.comb(n, k) - 1) - idx
    for _ in range(0, k):
        scored = ((x, get_below(x)) for x in reversed(range(n_p)))
        n_p, below = next(x for x in scored if x[1] <= above)
        result.append(n - 1 - n_p)
        above -= below
        k_p -= 1
    return result


def _pick_sp500_symbols(choice: int) -> list[str]:
    """Pick a subset of the S&P 500 symbols.

    :param choice: [0, comb(n_symbols, N_PICKS)) - The choice of subset.
    :return: A list of symbols.
    """
    return [SP_500[i] for i in _get_ith_comb(len(SP_500), N_PICKS, choice)]


# =============================================================================
#
# Generate a 49-bit integer from an image.
#
# =============================================================================


def _get_bin_str_from_image(path_to_image: str) -> str:
    """Get a binary string from an image.

    :param path_to_image: The path to the image.
    :return: A binary string inferred from the image.
    """
    mono_image = Image.open(path_to_image).convert("L")
    rows_mono_image = mono_image.resize((SR_BITS, 1))
    cols_mono_image = mono_image.resize((1, SR_BITS))
    average_l = np.array(mono_image.resize((1, 1)))[0][0]

    rows = (np.array(rows_mono_image) > average_l).flatten()
    cols = (np.array(cols_mono_image) > average_l).flatten()

    return "0b" + "".join(str(int(r == c)) for r, c in it.product(rows, cols))


def _interp_49_bit(x: str, y_sup: int) -> int:
    """Interpolate x in ['0b0', 2**49) to [0, y_sup).

    :param x: The binary value to interpolate.
    :param y_sup: the least upper bound of the output values.
    :return: The interpolated integer value [0, y_sup).
    """
    return math.floor(int(x, 2) / 2 ** (SR_BITS**2) * y_sup)


# =============================================================================
#
# Select a subset of the S&P 500 from an image.
#
# =============================================================================


def _get_number_from_image(path_to_image: str, num_choices: int) -> int:
    """Convert an image to an integer [0, num_choices).

    :param path_to_image: The image to convert.
    :param num_choices: The number of choices to choose from
    :return: The number represented by the image, from 0 to num_choices - 1.
    """
    as_bin = _get_bin_str_from_image(path_to_image)
    return _interp_49_bit(as_bin, num_choices)


def _pick_stocks(path_to_image: str) -> list[str]:
    """Pick N_PICKS stocks from the S&P 500.

    :param path_to_image: The image to use to pick stocks.
    :return: A list of N_PICKS stock symbols.
    """
    num_choices = math.comb(len(SP_500), N_PICKS)
    choice = _get_number_from_image(path_to_image, num_choices)
    return _pick_sp500_symbols(choice)


# =============================================================================
#
# Create a parser and run script
#
# =============================================================================


def _get_parser() -> argparse.ArgumentParser:
    """Return an argument parser that accepts one argument, a path to an image file.

    :return: An argument parser
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("image", nargs="?")
    return parser


def _main() -> None:
    """Run the script."""
    parser = _get_parser()
    args = parser.parse_args()
    print(_pick_stocks(args.image or input("Enter path to image: ")))
    _ = os.system("pause")


if __name__ == "__main__":
    _main()
