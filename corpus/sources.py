"""RELATE v0.1 source tables — entity families and structured facts.

Stage 4 of the build sequence (spec §15). Every fact here is either invented for
RELATE (`authored`) or a widely-known public fact restated in original wording
(`adapted`, with an as-of note). No text is lifted from an existing NLI / STS /
retrieval corpus — items are generated from these tables by `generate.py`, which
is the corpus's primary contamination defense (spec §8).

An *entity family* is a set of same-type entities that share a fact schema, so a
`split_entity` fold can hold whole families out. A family's `kind` selects which
template families (`templates.py`) render it and which typed pairs are known by
construction.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# geo-civics — capitals, neighbours, landlocked status, populations over time
# adapted public facts (as of 2024); wrong-answer siblings are real same-type
# cities so `contradiction` pairs are defensible.
# --------------------------------------------------------------------------- #
GEO_CAPITALS = [
    # (family_slug, country, capital, wrong_capital, region, a_second_city)
    ("ireland", "Ireland", "Dublin", "Cork", "Western Europe", "Galway"),
    ("france", "France", "Paris", "Lyon", "Western Europe", "Marseille"),
    ("portugal", "Portugal", "Lisbon", "Porto", "Southern Europe", "Braga"),
    ("spain", "Spain", "Madrid", "Barcelona", "Southern Europe", "Seville"),
    ("norway", "Norway", "Oslo", "Bergen", "Northern Europe", "Trondheim"),
    ("finland", "Finland", "Helsinki", "Tampere", "Northern Europe", "Turku"),
    ("greece", "Greece", "Athens", "Thessaloniki", "Southern Europe", "Patras"),
    ("austria", "Austria", "Vienna", "Graz", "Central Europe", "Linz"),
    ("poland", "Poland", "Warsaw", "Krakow", "Central Europe", "Lodz"),
    ("hungary", "Hungary", "Budapest", "Debrecen", "Central Europe", "Szeged"),
    ("romania", "Romania", "Bucharest", "Cluj-Napoca", "Eastern Europe", "Timisoara"),
    ("bulgaria", "Bulgaria", "Sofia", "Plovdiv", "Eastern Europe", "Varna"),
    ("morocco", "Morocco", "Rabat", "Casablanca", "North Africa", "Fez"),
    ("kenya", "Kenya", "Nairobi", "Mombasa", "East Africa", "Kisumu"),
    ("ghana", "Ghana", "Accra", "Kumasi", "West Africa", "Tamale"),
    ("vietnam", "Vietnam", "Hanoi", "Ho Chi Minh City", "Southeast Asia", "Da Nang"),
    ("thailand", "Thailand", "Bangkok", "Chiang Mai", "Southeast Asia", "Phuket"),
    ("chile", "Chile", "Santiago", "Valparaiso", "South America", "Concepcion"),
    ("peru", "Peru", "Lima", "Arequipa", "South America", "Cusco"),
    ("ecuador", "Ecuador", "Quito", "Guayaquil", "South America", "Cuenca"),
    ("canada", "Canada", "Ottawa", "Toronto", "North America", "Montreal"),
    ("australia", "Australia", "Canberra", "Sydney", "Oceania", "Melbourne"),
    ("newzealand", "New Zealand", "Wellington", "Auckland", "Oceania", "Christchurch"),
    ("turkey", "Turkey", "Ankara", "Istanbul", "Western Asia", "Izmir"),
]

# (family_slug, country, landlocked: bool, a neighbour, a body of water it does NOT touch)
GEO_LANDLOCKED = [
    ("bolivia", "Bolivia", True, "Paraguay", "Pacific Ocean"),
    ("paraguay", "Paraguay", True, "Bolivia", "Atlantic Ocean"),
    ("nepal", "Nepal", True, "India", "Bay of Bengal"),
    ("austria2", "Austria", True, "Slovenia", "Mediterranean Sea"),
    ("hungary2", "Hungary", True, "Serbia", "Adriatic Sea"),
    ("mongolia", "Mongolia", True, "Kazakhstan", "Yellow Sea"),
    ("zambia", "Zambia", True, "Angola", "Indian Ocean"),
    ("mali", "Mali", True, "Senegal", "Atlantic Ocean"),
    ("portugal2", "Portugal", False, "Spain", None),
    ("chile2", "Chile", False, "Argentina", None),
    ("vietnam2", "Vietnam", False, "Laos", None),
    ("kenya2", "Kenya", False, "Uganda", None),
]

# (family_slug, place, [(year, population)], is a city)
GEO_POP = [
    ("rivertown", "Rivertown", [(1990, "240,000"), (2005, "410,000"), (2020, "610,000")]),
    ("harborcity", "Harbor City", [(1995, "1.1 million"), (2010, "1.6 million"), (2023, "2.0 million")]),
    ("mesa-junction", "Mesa Junction", [(2000, "58,000"), (2012, "96,000"), (2022, "128,000")]),
    ("oldport", "Oldport", [(1985, "512,000"), (2001, "489,000"), (2019, "455,000")]),
    ("greenfield", "Greenfield", [(1998, "77,000"), (2011, "150,000"), (2024, "205,000")]),
    ("saltmarsh", "Saltmarsh", [(1992, "33,000"), (2008, "41,000"), (2021, "39,000")]),
]

# (family_slug, country, parliament chambers, lower-house seats, an org joined, year joined)
GEO_CIVICS = [
    ("lindenia", "Lindenia", 2, 300, "the regional trade bloc", 1996),
    ("vorland", "Vorland", 1, 165, "the monetary union", 2004),
    ("acaria", "Acaria", 2, 448, "the security pact", 1991),
    ("belmarck", "Belmarck", 1, 120, "the customs area", 2011),
    ("carneth", "Carneth", 2, 350, "the free-travel zone", 2007),
]

# --------------------------------------------------------------------------- #
# corporate-events — acquisitions (role order matters), headcount over time, CEOs
# invented companies; `authored`.
# --------------------------------------------------------------------------- #
CORP_ACQ = [
    # (family_slug, acquirer, target, year, structure, sector)
    ("acme-beta", "Acme Corp", "Beta Systems", 2019, "an all-cash deal", "industrial software"),
    ("northwind-clearwater", "Northwind Logistics", "Clearwater Freight", 2021, "a stock-and-cash transaction", "logistics"),
    ("helioss-pinebank", "Helios Semiconductor", "Pine Analog", 2020, "an all-cash tender offer", "semiconductors"),
    ("gridpoint-voltaic", "GridPoint Energy", "Voltaic Storage", 2022, "a share exchange", "grid storage"),
    ("meridian-quill", "Meridian Media", "Quill Publishing", 2018, "a cash acquisition", "publishing"),
    ("cobalt-riverside", "Cobalt Foods", "Riverside Dairy", 2023, "an asset purchase", "food processing"),
    ("stratus-lumen", "Stratus Cloud", "Lumen Metrics", 2021, "an all-stock merger", "observability software"),
    ("keystone-fairwind", "Keystone Rail", "Fairwind Coaches", 2017, "a cash-and-debt deal", "passenger transport"),
    ("aurora-bluefin", "Aurora Robotics", "Bluefin Marine", 2024, "an all-cash deal", "autonomous systems"),
    ("verdant-oakline", "Verdant Pharma", "Oakline Biotech", 2019, "a cash-plus-milestones deal", "pharmaceuticals"),
]

# (family_slug, firm, [(year, headcount)], a governance event, its year)
CORP_HEAD = [
    ("northwind", "Northwind Logistics", [(2015, "1,200"), (2020, "4,600"), (2024, "8,500")], "named a new chief executive", 2022),
    ("helios", "Helios Semiconductor", [(2012, "800"), (2018, "3,100"), (2023, "5,400")], "moved its headquarters", 2021),
    ("meridian", "Meridian Media", [(2010, "6,000"), (2017, "4,200"), (2024, "3,100")], "spun off its events division", 2020),
    ("cobalt", "Cobalt Foods", [(2009, "2,400"), (2016, "5,800"), (2023, "7,000")], "closed two plants", 2019),
    ("stratus", "Stratus Cloud", [(2016, "150"), (2020, "900"), (2024, "2,300")], "held its first investor day", 2023),
    ("keystone", "Keystone Rail", [(2011, "9,500"), (2018, "8,100"), (2024, "7,700")], "renegotiated its union contract", 2022),
]

# --------------------------------------------------------------------------- #
# biomed-claims — multi-effect claims (for partial-support), effect over time,
# populations. invented drugs / compounds; `authored`. Deliberately generic so
# no real trial is implied.
# --------------------------------------------------------------------------- #
BIO_MULTI = [
    # (family_slug, agent, effect1, effect2, population, condition)
    ("drugx-elderly", "drug X", "lowered blood pressure", "improved sleep quality", "elderly patients", "hypertension"),
    ("compz-adults", "compound Z", "reduced joint pain", "increased walking distance", "adults with osteoarthritis", "osteoarthritis"),
    ("agentq-children", "agent Q", "shortened fever duration", "reduced cough frequency", "children", "viral respiratory infection"),
    ("serumk-women", "serum K", "raised haemoglobin", "reduced fatigue scores", "women with anaemia", "iron-deficiency anaemia"),
    ("inhibm-smokers", "inhibitor M", "cut relapse rates", "eased withdrawal symptoms", "adult smokers", "nicotine dependence"),
    ("peptr-diabetics", "peptide R", "lowered fasting glucose", "reduced weight", "adults with type 2 diabetes", "type 2 diabetes"),
    ("saltn-athletes", "formulation N", "delayed muscle fatigue", "improved recovery time", "endurance athletes", "exercise fatigue"),
    ("creamp-patients", "cream P", "reduced lesion count", "decreased itching", "patients with eczema", "atopic dermatitis"),
]

# (family_slug, study subject, effect, [(year, verdict)], effect-size)
BIO_TIME = [
    ("cohort-trial", "the treatment arm", "a relative risk reduction", [(2011, "a 12% relative risk reduction"), (2019, "no significant risk reduction")], "12%"),
    ("statin-review", "the intervention group", "a change in cholesterol", [(2008, "a large LDL reduction"), (2020, "a modest LDL reduction")], "modest"),
    ("vaccine-followup", "the vaccinated cohort", "protection against infection", [(2021, "88% efficacy"), (2023, "54% efficacy")], "waned"),
    ("supplement-arm", "the supplement group", "an effect on bone density", [(2013, "a significant gain"), (2022, "no measurable gain")], "none"),
]

# --------------------------------------------------------------------------- #
# product-support — the product itself is the "named entity" for this domain
# (spec §17.2). informal user register; `authored`.
# --------------------------------------------------------------------------- #
PROD = [
    # (family_slug, product, fault, condition, version, feature, feature_location)
    ("printer-jam", "the printer", "jams", "on thick card stock", "the latest firmware", "the duplex setting", "the paper-handling menu"),
    ("app-crash", "the app", "crashes on launch", "after the 4.2 update", "4.2", "the CSV export", "the reports screen"),
    ("router-drop", "the router", "drops the connection", "every few minutes on 5GHz", "the 3.0 firmware", "the guest network toggle", "the wireless settings page"),
    ("watch-sync", "the watch", "fails to sync", "when Bluetooth is left on overnight", "watchOS 9", "the sleep-tracking screen", "the health tab"),
    ("laptop-fan", "the laptop", "runs the fan constantly", "even when idle on battery", "the 2.4 BIOS", "the battery-saver profile", "the power settings"),
    ("camera-focus", "the camera", "hunts for focus", "in low light with the kit lens", "firmware 1.6", "back-button focus", "the custom-controls menu"),
    ("speaker-pair", "the speaker", "will not pair", "with more than one phone at a time", "the 5.1 app", "the multi-room group", "the devices screen"),
    ("thermostat-schedule", "the thermostat", "ignores the schedule", "on weekends", "the 2.2 app", "the vacation hold", "the schedule tab"),
]

# --------------------------------------------------------------------------- #
# everyday-statements — high lexical overlap negation / paraphrase stress cases
# `authored`.
# --------------------------------------------------------------------------- #
EVERYDAY = [
    # (family_slug, subject, verb_past, object, adverb, passive_form)
    ("committee-proposal", "the committee", "rejected", "the proposal", "unanimously", "the proposal was turned down by the committee"),
    ("board-budget", "the board", "approved", "the budget", "without amendment", "the budget was signed off by the board"),
    ("editor-article", "the editor", "spiked", "the article", "at the last minute", "the article was pulled by the editor"),
    ("panel-application", "the panel", "shortlisted", "the application", "on the first round", "the application was advanced by the panel"),
    ("landlord-request", "the landlord", "declined", "the repair request", "in writing", "the repair request was refused by the landlord"),
    ("council-permit", "the council", "granted", "the permit", "after a hearing", "the permit was issued by the council"),
    ("teacher-extension", "the teacher", "allowed", "the extension", "for medical reasons", "the extension was given by the teacher"),
    ("jury-claim", "the jury", "dismissed", "the claim", "after an hour", "the claim was thrown out by the jury"),
]

# everyday scenes for entailment (specific -> general) and simple states
EVERYDAY_SCENES = [
    # (family_slug, specific_subject, state, location, general_subject)
    ("porch-dog", "a golden retriever", "is asleep", "on the porch", "a dog"),
    ("kitchen-cat", "a tabby cat", "is sitting", "on the kitchen counter", "a cat"),
    ("garage-sedan", "a red sedan", "is parked", "in the garage", "a car"),
    ("desk-laptop", "an open laptop", "is charging", "on the desk", "a computer"),
    ("hall-bicycle", "a mountain bike", "is leaning", "against the hallway wall", "a bicycle"),
    ("pond-swan", "a black swan", "is gliding", "across the pond", "a bird"),
]

# unrelated filler items, one per domain, cross-paired only
FILLER = [
    ("weekend-weather", "everyday-statements", "It rained for most of the weekend."),
    ("bus-timetable", "everyday-statements", "The last bus leaves at eleven on weeknights."),
    ("museum-hours", "geo-civics", "The city museum is closed on Mondays."),
    ("recipe-flour", "everyday-statements", "The recipe calls for two cups of flour."),
    ("garden-frost", "everyday-statements", "An early frost damaged the tomato plants."),
    ("kettle-boil", "everyday-statements", "The kettle switched itself off once it boiled."),
    ("piano-tune", "everyday-statements", "The piano was tuned before the recital."),
    ("ferry-fog", "everyday-statements", "The morning ferry was cancelled because of fog."),
]

# --------------------------------------------------------------------------- #
# competition / adjudication events -> role-order matters (relation-swap rich)
# invented; `authored`.
# --------------------------------------------------------------------------- #
COMPETE = [
    # (family_slug, winner, loser, event, margin, domain_register)
    ("league-final", "Redcliff United", "Marden Town", "the league final", "2-1", "corporate-events"),
    ("appeal-ruling", "the tenants' association", "the property firm", "the planning appeal", "on all three grounds", "corporate-events"),
    ("bid-contest", "Harlow Rail", "Coastline Transit", "the franchise bid", "on cost and reliability scores", "corporate-events"),
    ("debate-round", "the Oxdown team", "the Ferris College team", "the semi-final debate", "by a single ballot", "everyday-statements"),
    ("tender-award", "Brightwork Ltd", "Summit Contracts", "the bridge-repair tender", "by four percent", "corporate-events"),
    ("court-case", "the plaintiff", "the defendant", "the breach-of-contract case", "with full costs", "corporate-events"),
    ("chess-match", "Petrova", "Alvarez", "the championship match", "6.5 to 5.5", "everyday-statements"),
    ("grant-round", "the Lindqvist lab", "the Osei lab", "the final grant round", "on impact score", "biomed-claims"),
]

# --------------------------------------------------------------------------- #
# more biomed multi-effect families (partial-support + negation volume)
# --------------------------------------------------------------------------- #
BIO_MULTI += [
    ("tabx-seniors", "tablet X", "reduced hospital readmissions", "improved mobility scores", "seniors after hip surgery", "post-operative recovery"),
    ("dropy-infants", "drop Y", "cleared the infection faster", "lowered fever", "infants with otitis media", "ear infection"),
    ("gelz-runners", "gel Z", "reduced swelling", "shortened return-to-play time", "runners with ankle sprains", "ankle sprain"),
    ("pilla-veterans", "pill A", "cut nightmare frequency", "improved sleep continuity", "veterans with PTSD", "post-traumatic stress"),
    ("sprayb-workers", "spray B", "relieved nasal congestion", "reduced sneezing", "workers with seasonal allergy", "allergic rhinitis"),
    ("patchc-menopause", "patch C", "reduced hot flushes", "improved mood scores", "women in menopause", "menopausal symptoms"),
]

# more product families
PROD += [
    ("tv-hdmi", "the television", "loses the HDMI signal", "when the soundbar is powered on", "the 8.1 update", "auto-input switching", "the input menu"),
    ("vacuum-dock", "the vacuum", "misses the dock", "when the rug is near the base", "firmware 2.0", "the no-go zone", "the map screen"),
    ("earbuds-anc", "the earbuds", "hiss with noise cancelling on", "during phone calls", "the 3.2 app", "transparency mode", "the earbud settings"),
    ("scale-weight", "the scale", "shows the wrong weight", "on carpet", "the 1.4 app", "the athlete mode", "the profile screen"),
    ("doorbell-motion", "the doorbell", "misses motion events", "at night", "the 4.0 firmware", "the pre-roll clip", "the video settings"),
    ("mouse-lag", "the mouse", "lags", "on wake from sleep", "driver 5.3", "the polling-rate setting", "the performance tab"),
]

# more everyday families
EVERYDAY += [
    ("union-offer", "the union", "accepted", "the pay offer", "after a ballot", "the pay offer was accepted by the union"),
    ("agency-brief", "the agency", "returned", "the brief", "with notes", "the brief was returned by the agency"),
    ("clinic-referral", "the clinic", "processed", "the referral", "within a week", "the referral was processed by the clinic"),
    ("society-motion", "the society", "tabled", "the motion", "until spring", "the motion was tabled by the society"),
]

# more everyday scenes
EVERYDAY_SCENES += [
    ("roof-pigeon", "a grey pigeon", "is perched", "on the roof", "a bird"),
    ("yard-oak", "a young oak", "is growing", "in the yard", "a tree"),
    ("shelf-novel", "a hardback novel", "is lying", "on the shelf", "a book"),
    ("lot-truck", "a delivery truck", "is idling", "in the lot", "a vehicle"),
]

# more corporate acquisitions
CORP_ACQ += [
    ("summit-delta", "Summit Analytics", "Delta Survey", 2020, "an all-cash deal", "market research"),
    ("orbit-harbour", "Orbit Telecom", "Harbour Wireless", 2022, "a stock swap", "telecommunications"),
    ("crest-lowland", "Crest Insurance", "Lowland Mutual", 2018, "a cash offer", "insurance"),
    ("pinnacle-brook", "Pinnacle Retail", "Brook Grocers", 2023, "an asset purchase", "grocery retail"),
    ("vector-ash", "Vector Aerospace", "Ash Components", 2021, "a cash-and-earnout deal", "aerospace parts"),
]

# more geo capitals
GEO_CAPITALS += [
    ("uruguay", "Uruguay", "Montevideo", "Salto", "South America", "Paysandu"),
    ("czechia", "Czechia", "Prague", "Brno", "Central Europe", "Ostrava"),
    ("slovakia", "Slovakia", "Bratislava", "Kosice", "Central Europe", "Presov"),
    ("croatia", "Croatia", "Zagreb", "Split", "Southern Europe", "Rijeka"),
    ("tunisia", "Tunisia", "Tunis", "Sfax", "North Africa", "Sousse"),
    ("jordan", "Jordan", "Amman", "Zarqa", "Western Asia", "Irbid"),
    ("senegal", "Senegal", "Dakar", "Touba", "West Africa", "Thies"),
    ("bolivia-c", "Bolivia", "Sucre", "La Paz", "South America", "Cochabamba"),
]
