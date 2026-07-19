"""Golden internal-consistency check of the TERRA split specification.

Pins every partition count from the spec against the authoritative sizing formulas in
``evals.split_spec`` -- no data, no split generation. This is the numerical oracle the generator's
realized manifests must later match. A change to the formulas or the golden tables that breaks the
accounting fails here.

Each row is re-derived from the frozen population + purge inputs:

    source (train,val,test) == source_partition_sizes(population - N_target - purged)
    target (pool,test)      == target_partition_sizes(N_target)
    population              == N_target + purged + (train+val+test)          [geographic LODO]
    cells partition the population exactly; official anchors sum to their totals.
"""

from __future__ import annotations

import pytest

from evals import split_spec as S

# --------------------------------------------------------------------------- #
# Golden expected partition sizes (verbatim from the specification tables)
# --------------------------------------------------------------------------- #
# geographic LODO rows: name -> (purged_source, (train,val,test), (pool,test))
CH_GEO = {
    "ethiopia":          (168, (53_354, 6_670, 6_670), (664, 166)),
    "ile-de-france":     (22,  (49_188, 6_149, 6_149), (4_947, 1_237)),
    "kenya":             (184, (51_601, 6_451, 6_451), (2_404, 601)),
    "lem-brazil":        (4,   (53_480, 6_686, 6_686), (668, 168)),
    "mali":              (29,  (53_897, 6_738, 6_738), (232, 58)),
    "martinique-france": (0,   (52_449, 6_557, 6_557), (1_703, 426)),
    "reunion-france":    (0,   (52_362, 6_546, 6_546), (1_790, 448)),
    "rwanda":            (49,  (50_789, 6_349, 6_349), (3_324, 832)),
    "sudan":             (163, (53_687, 6_712, 6_712), (334, 84)),
    "togo":              (64,  (52_840, 6_605, 6_605), (1_262, 316)),
    "usa-kern":          (18,  (45_193, 5_650, 5_650), (8_944, 2_237)),
}
# one-class supplementary source-only targets: name -> (target_examples, purged, (train,val,test))
CH_ONECLASS = {
    "central-asia": (4_893, 49, (50_200, 6_275, 6_275)),
    "tanzania":     (392,   71, (53_783, 6_723, 6_723)),
    "uganda":       (233,   13, (53_956, 6_745, 6_745)),
    "zimbabwe":     (49,    55, (54_070, 6_759, 6_759)),
}
# spatial cells: id -> (cell_total, (train,val,test), (pool,test))
CH_CELLS = {
    0: (8_970,  (46_973, 5_872, 5_872), (7_176, 1_794)),
    1: (18_433, (39_380, 4_923, 4_923), (14_746, 3_687)),
    2: (13_181, (43_592, 5_449, 5_449), (10_544, 2_637)),
    3: (14_657, (42_423, 5_303, 5_303), (11_725, 2_932)),
    4: (12_451, (44_151, 5_520, 5_520), (9_960, 2_491)),
}

EU_GEO = {
    "Estonia":  (29_936, (400_665, 50_084, 50_084), (140_713, 35_179)),
    "Latvia":   (24_895, (200_502, 25_063, 25_063), (344_910, 86_228)),
    "Portugal": (0,      (485_624, 60_703, 60_703), (79_704, 19_927)),
}
EU_CELLS = {
    0: (51_954,  (522_194, 65_275, 65_275), (41_563, 10_391)),
    1: (47_677,  (524_079, 65_510, 65_510), (38_141, 9_536)),
    2: (273_548, (312_698, 39_088, 39_088), (218_838, 54_710)),
    3: (135_297, (436_472, 54_560, 54_560), (108_237, 27_060)),
    4: (198_185, (379_799, 47_475, 47_475), (158_548, 39_637)),
}

BZ_GEO = {
    "frh01": (30_942, (318_966, 39_871, 39_871), (142_890, 35_723)),
    "frh02": (15_509, (361_687, 45_211, 45_211), (112_516, 28_129)),
    "frh03": (18_710, (338_528, 42_317, 42_317), (133_112, 33_279)),
    "frh04": (29_170, (365_183, 45_648, 45_648), (98_091, 24_523)),
}
BZ_CELLS = {
    0: (113_228, (382_837, 47_855, 47_855), (90_582, 22_646)),
    1: (132_021, (369_342, 46_168, 46_168), (105_616, 26_405)),
    2: (129_352, (375_189, 46_899, 46_899), (103_481, 25_871)),
    3: (110_105, (380_791, 47_600, 47_600), (88_084, 22_021)),
    4: (123_557, (372_447, 46_557, 46_557), (98_845, 24_712)),
}

PA_GEO = {
    "T30UXV": (0, (1_520, 191, 191), (424, 107)),
    "T31TFJ": (0, (1_448, 181, 181), (498, 125)),
    "T31TFM": (0, (1_368, 171, 171), (578, 145)),
    "T32ULU": (0, (1_501, 188, 188), (444, 112)),
}
PA_CELLS = {
    0: (623, (1_448, 181, 181), (498, 125)),
    1: (367, (1_652, 207, 207), (293, 74)),
    2: (356, (1_661, 208, 208), (284, 72)),
    3: (556, (1_501, 188, 188), (444, 112)),
    4: (531, (1_520, 191, 191), (424, 107)),
}

# random_id: benchmark -> (train, val, test)
RANDOM_ID = {
    "cropharvest": (54_152, 6_770, 6_770),
    "eurocropsml": (565_327, 70_667, 70_667),
    "breizhcrops": (486_609, 60_827, 60_827),
    "pastis":      (1_945, 244, 244),
}


# --------------------------------------------------------------------------- #
# Formula-level unit checks
# --------------------------------------------------------------------------- #
def test_source_formula_is_exact_80_10_10():
    train, val, test = S.source_partition_sizes(67_692)
    assert (val, test) == (6_770, 6_770) and train == 54_152
    assert train + val + test == 67_692


def test_target_formula_is_exact_80_20():
    assert S.target_partition_sizes(830) == (664, 166)
    assert S.target_partition_sizes(11_181) == (8_944, 2_237)  # ceil(0.2*11181)=2237


def test_official_formula_is_90_10_floor():
    # CropHarvest Togo provenance 1,272 -> 1,145 / 127
    assert S.official_source_train_val_sizes(1_272) == (1_145, 127)


def test_source_formula_rejects_degenerate_population():
    with pytest.raises(ValueError):
        S.source_partition_sizes(2)  # ceil(0.1*2)=1 twice -> train 0


# --------------------------------------------------------------------------- #
# Table-level golden consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bench,sizes", RANDOM_ID.items())
def test_random_id_rows(bench, sizes):
    pop = S.ALL_SPECS[bench].population
    train, val, test = sizes
    assert S.source_partition_sizes(pop) == (train, val, test)
    assert train + val + test == pop


def _check_geo(population, table):
    for name, (purged, src, tgt) in table.items():
        train, val, test = src
        pool, ttest = tgt
        n_target = pool + ttest
        assert S.source_partition_sizes(population - n_target - purged) == (train, val, test), name
        assert S.target_partition_sizes(n_target) == (pool, ttest), name
        assert population == n_target + purged + (train + val + test), name


def test_cropharvest_geographic_rows():
    _check_geo(S.CROPHARVEST.population, CH_GEO)
    assert len(CH_GEO) == len(S.CROPHARVEST.geographic_targets) == 11
    assert set(CH_GEO) == set(S.CROPHARVEST.geographic_targets)


def test_cropharvest_oneclass_rows():
    pop = S.CROPHARVEST.population
    for name, (tex, purged, src) in CH_ONECLASS.items():
        assert S.source_partition_sizes(pop - tex - purged) == src, name
    assert set(CH_ONECLASS) == set(S.CROPHARVEST.supplementary_targets)


def test_cropharvest_official_row():
    o = S.CROPHARVEST_OFFICIAL
    assert S.official_source_train_val_sizes(o["source_pool"]) == (o["source_train"], o["source_val"]) == (1_145, 127)
    # filled release class composition (no None placeholders); each subdivision's classes sum
    tr, va, te = o["source_train_classes"], o["source_val_classes"], o["eval_test_classes"]
    assert tr["crop"] + tr["non_crop"] == o["source_train"] == 1_145
    assert va["crop"] + va["non_crop"] == o["source_val"] == 127
    assert te["crop"] + te["non_crop"] == o["eval_test"] == 306
    # source pool class counts == train + val class counts, and sum to the pool
    pc = o["source_pool_classes"]
    assert pc["crop"] == tr["crop"] + va["crop"] and pc["non_crop"] == tr["non_crop"] + va["non_crop"]
    assert pc["crop"] + pc["non_crop"] == o["source_pool"] == 1_272
    assert None not in pc.values()  # placeholders filled


def test_eurocropsml_geographic_rows():
    _check_geo(S.EUROCROPML.population, EU_GEO)
    # target totals equal the country freezes
    for country, (_purged, _src, (pool, test)) in EU_GEO.items():
        assert pool + test == S.EUROCROPS_COUNTRY_POPULATION[country], country


def test_eurocropsml_population_freeze():
    assert S.EUROCROPS_RAW_POPULATION - sum(S.EUROCROPS_REMOVED_CLASSES.values()) == S.EUROCROPML.population
    assert sum(S.EUROCROPS_COUNTRY_POPULATION.values()) == S.EUROCROPML.population
    assert len(S.EUROCROPS_REMOVED_CLASSES) == 7
    assert sum(S.EUROCROPS_REMOVED_CLASSES.values()) == 22


def test_eurocropsml_official_anchor_test_is_estonia_020():
    _pool, ee_test = S.target_partition_sizes(S.EUROCROPS_COUNTRY_POPULATION["Estonia"])
    assert ee_test == 35_179
    assert S.EUROCROPS_OFFICIAL["Latvia->Estonia"]["test"] == 35_179
    assert S.EUROCROPS_OFFICIAL["Latvia+Portugal->Estonia"]["test"] == 35_179


def test_breizhcrops_geographic_rows():
    _check_geo(S.BREIZHCROPS.population, BZ_GEO)
    assert set(BZ_GEO) == set(S.BREIZHCROPS.geographic_targets)


def test_breizhcrops_official_anchor():
    o = S.BREIZHCROPS_OFFICIAL
    assert o["train"] + o["val"] + o["test"] == S.BREIZHCROPS.population


def test_pastis_geographic_rows():
    _check_geo(S.PASTIS.population, PA_GEO)
    # tile target totals equal the tile patch counts; purge is 0 everywhere
    for tile, (purged, _src, (pool, test)) in PA_GEO.items():
        assert purged == 0
        assert pool + test == S.PASTIS_TILE_PATCHES[tile], tile


def test_pastis_population_and_official():
    assert sum(S.PASTIS_TILE_PATCHES.values()) == S.PASTIS.population == 2_433
    o = S.PASTIS_OFFICIAL
    assert o["train"] + o["val"] + o["test"] == S.PASTIS.population


# --------------------------------------------------------------------------- #
# Spatial cells: internal consistency + partition-of-population
# --------------------------------------------------------------------------- #
def _check_cells(population, cells, *, purge_zero=False):
    total = 0
    for cid, (cell_total, src, tgt) in cells.items():
        train, val, test = src
        pool, ttest = tgt
        assert S.source_partition_sizes(train + val + test) == (train, val, test), cid
        assert S.target_partition_sizes(cell_total) == (pool, ttest), cid
        assert pool + ttest == cell_total, cid
        implied_purge = population - cell_total - (train + val + test)
        assert implied_purge >= 0, (cid, implied_purge)
        if purge_zero:
            assert implied_purge == 0, (cid, implied_purge)
        total += cell_total
    assert total == population, (total, population)


def test_cropharvest_cells_partition_population():
    _check_cells(S.CROPHARVEST.population, CH_CELLS)


def test_eurocropsml_cells_partition_population():
    _check_cells(S.EUROCROPML.population, EU_CELLS)
    assert 268_546 + 5_002 == EU_CELLS[2][0]
    assert 128_806 + 6_491 == EU_CELLS[3][0]
    assert 164_399 + 33_786 == EU_CELLS[4][0]


def test_breizhcrops_cells_partition_population():
    _check_cells(S.BREIZHCROPS.population, BZ_CELLS)
    assert 100_819 + 8_773 + 2_919 + 717 == BZ_CELLS[0][0]


def test_pastis_cells_partition_population():
    _check_cells(S.PASTIS.population, PA_CELLS, purge_zero=True)
    assert 367 + 356 == S.PASTIS_TILE_PATCHES["T31TFM"]  # T31TFM split into south + north


def test_number_of_cells_matches_config():
    for cells in (CH_CELLS, EU_CELLS, BZ_CELLS, PA_CELLS):
        assert len(cells) == S.N_CLUSTERS == 5
