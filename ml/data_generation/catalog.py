"""Static catalogues for the fictional chocolate & malt-beverage ecosystem.

Everything in this file is SYNTHETIC / ILLUSTRATIVE. Prices are illustrative INR
values chosen to be of a plausible order of magnitude for a demonstration; they
are NOT verified market quotes or company figures.
"""

# region: (country, lat, lon, base_geo_risk, monsoon_exposure, cyclone_exposure, import_lead_extra_days)
REGIONS = {
    "Maharashtra": ("India", 19.7, 75.7, 0.30, 0.8, 0.3, 0),
    "Gujarat": ("India", 22.3, 71.2, 0.30, 0.5, 0.6, 0),
    "Karnataka": ("India", 15.3, 75.7, 0.25, 0.6, 0.1, 0),
    "Kerala": ("India", 10.85, 76.27, 0.40, 1.0, 0.3, 0),
    "Tamil Nadu": ("India", 11.1, 78.7, 0.35, 0.5, 0.8, 0),
    "Andhra Pradesh": ("India", 15.9, 79.7, 0.35, 0.6, 0.8, 0),
    "West Bengal": ("India", 22.98, 87.85, 0.40, 0.9, 0.9, 0),
    "Uttar Pradesh": ("India", 26.8, 80.9, 0.30, 0.6, 0.0, 0),
    "Punjab": ("India", 31.1, 75.3, 0.20, 0.4, 0.0, 0),
    "Madhya Pradesh": ("India", 22.97, 78.65, 0.25, 0.6, 0.0, 0),
    "Odisha": ("India", 20.95, 85.1, 0.45, 0.8, 1.0, 0),
    "Haryana": ("India", 29.06, 76.09, 0.20, 0.4, 0.0, 0),
    "Goa": ("India", 15.3, 74.1, 0.30, 0.9, 0.2, 0),
    "Cote d'Ivoire": ("Cote d'Ivoire", 7.54, -5.55, 0.55, 0.7, 0.0, 35),
    "Ghana": ("Ghana", 7.95, -1.02, 0.50, 0.7, 0.0, 35),
    "Indonesia": ("Indonesia", -0.79, 113.9, 0.45, 0.8, 0.2, 20),
    "Malaysia": ("Malaysia", 4.21, 101.98, 0.35, 0.7, 0.1, 18),
    "Australia": ("Australia", -25.3, 133.8, 0.25, 0.2, 0.3, 25),
    "New Zealand": ("New Zealand", -40.9, 174.9, 0.20, 0.1, 0.1, 28),
    "China": ("China", 35.9, 104.2, 0.35, 0.4, 0.3, 22),
    "Netherlands": ("Netherlands", 52.1, 5.3, 0.15, 0.1, 0.0, 30),
}

# Monthly seasonal weather/logistics pattern (index 0 = Jan). Monsoon peaks Jun-Sep in India;
# West-African harmattan / mid-crop lean season Dec-Feb; NE monsoon/cyclone Oct-Dec on east coast.
MONSOON = [0.0, 0.0, 0.0, 0.05, 0.2, 0.7, 1.0, 1.0, 0.7, 0.2, 0.05, 0.0]
CYCLONE = [0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.1, 0.4, 0.8, 0.6]
WEST_AFRICA = [0.7, 0.6, 0.2, 0.1, 0.2, 0.4, 0.5, 0.4, 0.2, 0.1, 0.3, 0.6]

# material_id: (name, category, level, unit, unit_price_inr, criticality, inputs, preferred_regions)
# level 1 = delivered to the manufacturer by Tier-1, level 2 = input to Tier-1 (supplied by Tier-2),
# level 3 = input to Tier-2 (supplied by Tier-3)
MATERIALS = {
    # ----- level 1 (28) -----
    "M01": ("Cocoa powder (natural)", "Cocoa", 1, "kg", 520, "A", ["M29"], ["Kerala", "Karnataka", "Indonesia", "Malaysia", "Ghana"]),
    "M02": ("Cocoa powder (alkalised)", "Cocoa", 1, "kg", 610, "A", ["M29"], ["Kerala", "Indonesia", "Malaysia", "Netherlands"]),
    "M03": ("Cocoa butter", "Cocoa", 1, "kg", 1450, "A", ["M29"], ["Indonesia", "Malaysia", "Cote d'Ivoire", "Ghana", "Netherlands"]),
    "M04": ("Cocoa liquor", "Cocoa", 1, "kg", 780, "A", ["M29"], ["Kerala", "Karnataka", "Cote d'Ivoire", "Ghana"]),
    "M05": ("Whole milk powder", "Dairy", 1, "kg", 360, "A", ["M30"], ["Gujarat", "Punjab", "Uttar Pradesh", "New Zealand"]),
    "M06": ("Skimmed milk powder", "Dairy", 1, "kg", 310, "A", ["M30"], ["Gujarat", "Maharashtra", "Punjab", "New Zealand"]),
    "M07": ("Whey powder", "Dairy", 1, "kg", 190, "B", ["M30"], ["Gujarat", "Netherlands", "New Zealand"]),
    "M08": ("Malt extract", "Malt", 1, "kg", 140, "A", ["M31"], ["Haryana", "Punjab", "Uttar Pradesh", "Australia"]),
    "M09": ("Malted barley flour", "Malt", 1, "kg", 95, "A", ["M31"], ["Haryana", "Punjab", "Madhya Pradesh"]),
    "M10": ("Refined sugar", "Sweetener", 1, "kg", 42, "A", ["M32"], ["Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu"]),
    "M11": ("Liquid glucose", "Sweetener", 1, "kg", 48, "B", ["M33"], ["Maharashtra", "Gujarat", "Karnataka"]),
    "M12": ("Maltodextrin", "Sweetener", 1, "kg", 62, "B", ["M33"], ["Maharashtra", "Gujarat", "Uttar Pradesh"]),
    "M13": ("Vegetable fat (palm-based)", "Fat", 1, "kg", 135, "A", ["M34"], ["Gujarat", "Andhra Pradesh", "Malaysia", "Indonesia"]),
    "M14": ("Soy lecithin", "Emulsifier", 1, "kg", 210, "B", ["M35"], ["Madhya Pradesh", "Maharashtra"]),
    "M15": ("PGPR emulsifier", "Emulsifier", 1, "kg", 640, "B", ["M34"], ["Malaysia", "Netherlands"]),
    "M16": ("Vanilla flavour", "Flavour", 1, "kg", 2900, "C", ["M33"], ["Maharashtra", "Karnataka", "Netherlands"]),
    "M17": ("Chocolate flavour", "Flavour", 1, "kg", 2400, "C", ["M33"], ["Maharashtra", "Netherlands"]),
    "M18": ("Vitamin-mineral premix", "Fortificant", 1, "kg", 3800, "A", ["M33"], ["Maharashtra", "Gujarat", "Netherlands"]),
    "M19": ("Hazelnut paste", "Nuts", 1, "kg", 980, "B", ["M36"], ["Maharashtra", "China"]),
    "M20": ("Almond pieces", "Nuts", 1, "kg", 860, "C", ["M36"], ["Punjab", "Australia"]),
    "M21": ("Wheat flour", "Cereal", 1, "kg", 34, "B", ["M37"], ["Punjab", "Haryana", "Madhya Pradesh", "Uttar Pradesh"]),
    "M22": ("Aluminium foil laminate", "Packaging", 1, "kg", 420, "A", ["M38", "M39"], ["Maharashtra", "Gujarat", "Odisha"]),
    "M23": ("Flexible laminate film", "Packaging", 1, "kg", 260, "A", ["M39"], ["Gujarat", "Maharashtra", "Haryana"]),
    "M24": ("PET jars", "Packaging", 1, "unit", 9, "A", ["M39"], ["Gujarat", "Maharashtra", "Tamil Nadu"]),
    "M25": ("Glass jars", "Packaging", 1, "unit", 14, "B", ["M40"], ["Uttar Pradesh", "Gujarat"]),
    "M26": ("Printed cartons", "Packaging", 1, "unit", 6, "B", ["M41"], ["Maharashtra", "Tamil Nadu", "West Bengal", "Goa"]),
    "M27": ("Corrugated shipper boxes", "Packaging", 1, "unit", 18, "B", ["M42"], ["Maharashtra", "Gujarat", "West Bengal", "Goa"]),
    "M28": ("Labels and caps", "Packaging", 1, "unit", 1.2, "C", ["M41", "M39"], ["Maharashtra", "Gujarat", "Goa"]),
    # ----- level 2 (14) -----
    "M29": ("Cocoa beans", "Agri commodity", 2, "kg", 380, "A", ["M43"], ["Cote d'Ivoire", "Ghana", "Kerala", "Karnataka", "Andhra Pradesh", "Indonesia"]),
    "M30": ("Raw milk", "Agri commodity", 2, "litre", 38, "A", ["M44"], ["Gujarat", "Punjab", "Uttar Pradesh", "Maharashtra", "Haryana"]),
    "M31": ("Malting barley", "Agri commodity", 2, "kg", 26, "A", ["M43", "M45"], ["Haryana", "Punjab", "Uttar Pradesh", "Australia"]),
    "M32": ("Sugarcane", "Agri commodity", 2, "tonne", 3400, "A", ["M43"], ["Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu"]),
    "M33": ("Maize starch", "Agri processed", 2, "kg", 36, "B", ["M45"], ["Maharashtra", "Gujarat", "Karnataka", "Madhya Pradesh"]),
    "M34": ("Crude palm oil", "Agri commodity", 2, "kg", 92, "A", ["M43"], ["Malaysia", "Indonesia", "Andhra Pradesh"]),
    "M35": ("Soybean", "Agri commodity", 2, "kg", 46, "B", ["M45"], ["Madhya Pradesh", "Maharashtra"]),
    "M36": ("Raw tree nuts", "Agri commodity", 2, "kg", 620, "C", ["M43"], ["Australia", "China", "Punjab"]),
    "M37": ("Wheat", "Agri commodity", 2, "kg", 24, "B", ["M43", "M45"], ["Punjab", "Haryana", "Madhya Pradesh", "Uttar Pradesh"]),
    "M38": ("Aluminium ingot", "Metal", 2, "kg", 230, "A", ["M46", "M50"], ["Odisha", "Madhya Pradesh", "China"]),
    "M39": ("Polymer resin (PE/PET)", "Petrochemical", 2, "kg", 115, "A", ["M47"], ["Gujarat", "Maharashtra", "China"]),
    "M40": ("Glass batch (sand and cullet)", "Mineral", 2, "kg", 12, "B", ["M49", "M50"], ["Uttar Pradesh", "Gujarat"]),
    "M41": ("Paper board", "Paper", 2, "kg", 78, "B", ["M48"], ["Tamil Nadu", "Maharashtra", "West Bengal", "Odisha"]),
    "M42": ("Kraft paper", "Paper", 2, "kg", 41, "B", ["M48"], ["Gujarat", "Maharashtra", "West Bengal", "Uttar Pradesh"]),
    # ----- level 3 (8) -----
    "M43": ("Fertiliser and agro-chemicals", "Agri input", 3, "kg", 28, "B", [], ["Gujarat", "Uttar Pradesh", "Odisha", "Andhra Pradesh", "China", "Cote d'Ivoire"]),
    "M44": ("Cattle feed", "Agri input", 3, "kg", 22, "B", [], ["Gujarat", "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra"]),
    "M45": ("Certified seeds", "Agri input", 3, "kg", 160, "C", [], ["Punjab", "Madhya Pradesh", "Karnataka", "Australia"]),
    "M46": ("Alumina", "Mineral", 3, "kg", 38, "B", [], ["Odisha", "Andhra Pradesh", "Australia"]),
    "M47": ("Naphtha feedstock", "Petrochemical", 3, "kg", 62, "A", [], ["Gujarat", "Maharashtra", "Malaysia"]),
    "M48": ("Wood pulp", "Forest product", 3, "kg", 55, "B", [], ["Tamil Nadu", "Odisha", "West Bengal", "Indonesia"]),
    "M49": ("Soda ash", "Chemical", 3, "kg", 31, "B", [], ["Gujarat", "Tamil Nadu"]),
    "M50": ("Industrial fuel (coal/gas)", "Energy", 3, "kg", 9, "B", [], ["Odisha", "West Bengal", "Madhya Pradesh", "Gujarat"]),
}

# Number of suppliers per material (Tier-1 materials skewed towards critical inputs; some single-sourced)
TIER1_COUNTS = {
    "M01": 6, "M02": 4, "M03": 5, "M04": 4, "M05": 6, "M06": 5, "M07": 3, "M08": 6, "M09": 4, "M10": 8,
    "M11": 3, "M12": 3, "M13": 5, "M14": 3, "M15": 1, "M16": 2, "M17": 2, "M18": 1, "M19": 2, "M20": 2,
    "M21": 4, "M22": 6, "M23": 6, "M24": 6, "M25": 2, "M26": 7, "M27": 7, "M28": 7,
}  # sum = 120
TIER2_COUNTS = {
    "M29": 24, "M30": 20, "M31": 16, "M32": 15, "M33": 18, "M34": 14, "M35": 9, "M36": 8, "M37": 12,
    "M38": 10, "M39": 22, "M40": 8, "M41": 13, "M42": 11,
}  # sum = 200
TIER3_COUNTS = {"M43": 40, "M44": 22, "M45": 22, "M46": 14, "M47": 24, "M48": 22, "M49": 14, "M50": 22}  # sum = 180

PRODUCT_FAMILIES = [
    # family, base_bom {material: kg or units per product unit}, unit (pack), price_inr per pack, margin share
    ("Milk chocolate bar", {"M01": 0.004, "M03": 0.012, "M04": 0.006, "M05": 0.010, "M10": 0.018, "M13": 0.002, "M14": 0.0004, "M15": 0.0001, "M16": 0.00003, "M22": 0.003, "M26": 1.0, "M27": 0.02, "M28": 0.0}, 40, 0.34),
    ("Dark chocolate bar", {"M02": 0.005, "M03": 0.014, "M04": 0.020, "M10": 0.012, "M14": 0.0004, "M16": 0.00002, "M22": 0.003, "M26": 1.0, "M27": 0.02}, 95, 0.40),
    ("Hazelnut chocolate", {"M01": 0.004, "M03": 0.010, "M05": 0.008, "M10": 0.014, "M19": 0.008, "M14": 0.0004, "M22": 0.003, "M26": 1.0, "M27": 0.02}, 110, 0.38),
    ("Malt beverage powder 500g jar", {"M08": 0.150, "M09": 0.080, "M01": 0.030, "M06": 0.090, "M10": 0.100, "M12": 0.030, "M18": 0.004, "M17": 0.0003, "M24": 1.0, "M28": 2.0, "M27": 0.05}, 265, 0.30),
    ("Malt beverage refill 1kg", {"M08": 0.300, "M09": 0.160, "M01": 0.060, "M06": 0.180, "M10": 0.200, "M12": 0.060, "M18": 0.008, "M17": 0.0006, "M23": 0.012, "M26": 1.0, "M27": 0.08}, 480, 0.28),
    ("Cocoa drink mix", {"M02": 0.060, "M10": 0.120, "M06": 0.050, "M11": 0.020, "M07": 0.030, "M17": 0.0005, "M23": 0.008, "M26": 1.0, "M27": 0.05}, 210, 0.31),
    ("Chocolate spread jar", {"M01": 0.025, "M10": 0.150, "M13": 0.090, "M19": 0.040, "M06": 0.030, "M14": 0.001, "M25": 1.0, "M28": 2.0, "M27": 0.05}, 330, 0.33),
    ("Chocolate-coated biscuit", {"M21": 0.050, "M01": 0.006, "M03": 0.006, "M10": 0.020, "M13": 0.010, "M11": 0.004, "M23": 0.003, "M26": 1.0, "M27": 0.02}, 30, 0.26),
    ("Malted milk biscuit", {"M21": 0.060, "M08": 0.010, "M05": 0.008, "M10": 0.018, "M13": 0.012, "M23": 0.003, "M26": 1.0, "M27": 0.02}, 25, 0.24),
    ("Almond chocolate bar", {"M01": 0.004, "M03": 0.012, "M05": 0.009, "M10": 0.016, "M20": 0.006, "M14": 0.0004, "M22": 0.003, "M26": 1.0, "M27": 0.02}, 120, 0.37),
]

SITES = [
    # site_id, name, type, region, city, lat, lon
    ("W01", "Plant P1 - Pune (synthetic)", "plant", "Maharashtra", "Pune", 18.52, 73.86),
    ("W02", "Plant P2 - Kolkata (synthetic)", "plant", "West Bengal", "Kolkata", 22.57, 88.36),
    ("W03", "Distribution centre DC-West (synthetic)", "distribution_centre", "Maharashtra", "Bhiwandi", 19.30, 73.06),
    ("W04", "Distribution centre DC-East (synthetic)", "distribution_centre", "West Bengal", "Howrah", 22.59, 88.26),
]

NAME_PREFIX = ["Sahyadri", "Konkan", "Deccan", "Malabar", "Narmada", "Ganga", "Kaveri", "Vindhya", "Aravali", "Coromandel",
               "Ashoka", "Shivalik", "Godavari", "Krishna", "Tapti", "Bharat", "Indus", "Nilgiri", "Satpura", "Sundarban",
               "Atlas", "Meridian", "Pacific", "Equator", "Harbour", "Summit", "Evergreen", "Golden", "Silverline", "Bluewater"]
NAME_SUFFIX = {
    "Cocoa": "Cocoa Processors", "Dairy": "Dairy Ingredients", "Malt": "Maltings", "Sweetener": "Sweeteners",
    "Fat": "Edible Fats", "Emulsifier": "Food Chemicals", "Flavour": "Flavours", "Fortificant": "Nutrition Premix",
    "Nuts": "Nut Processors", "Cereal": "Flour Mills", "Packaging": "Packaging", "Agri commodity": "Agro Traders",
    "Agri processed": "Agro Processing", "Metal": "Metals", "Petrochemical": "Polymers", "Mineral": "Minerals",
    "Paper": "Paper Mills", "Agri input": "Agri Inputs", "Forest product": "Pulp Industries", "Chemical": "Chemicals",
    "Energy": "Energy Supply",
}
