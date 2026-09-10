"""RELATE-DEV source — hand-authored, every pair meant to be manually inspected.

Purpose (spec build step 3): a tiny set (a few examples per relation across
domains) whose only job is to surface ONTOLOGY DEFECTS while corrections are
cheap. Not for measurement. build_dev.py turns this into items/pairs/queries
JSONL, computes the lexical/entity overlap fields, and runs the validators.

Authoring rules followed here:
  - every item's `entities` list is exact, so entity_overlap auto-checks pass
  - `atomic_claims` filled for any item used as the `a` side of partial-support
  - `time_index` set for both items of every temporal-mismatch pair
  - negation pairs are minimal edits (overlap will be >= 0.7)
  - `runner_up` records the next-best relation per spec 3.9
"""

# --------------------------------------------------------------------------- #
# ITEMS
# --------------------------------------------------------------------------- #
# fields: id, text, domain, template_family, entity_family, entities,
#         [time_index], [atomic_claims], provenance{kind,...}
ITEMS = [
    # --- geo-civics -------------------------------------------------------- #
    dict(id="i-geo001", text="Dublin is the capital of Ireland.",
         domain="geo-civics", template_family="tf-x-is-capital-of-y", entity_family="ef-ireland",
         entities=["Dublin", "Ireland"], provenance=dict(kind="authored")),
    dict(id="i-geo002", text="The capital of Ireland is Dublin.",
         domain="geo-civics", template_family="tf-capital-of-y-is-x", entity_family="ef-ireland",
         entities=["Dublin", "Ireland"], provenance=dict(kind="authored")),
    dict(id="i-geo003", text="Dublin is not the capital of Ireland.",
         domain="geo-civics", template_family="tf-x-is-capital-of-y", entity_family="ef-ireland",
         entities=["Dublin", "Ireland"],
         provenance=dict(kind="perturbed", parent_id="i-geo001", transformation="negation")),
    dict(id="i-geo004", text="Ireland's largest city serves as its seat of government.",
         domain="geo-civics", template_family="tf-largest-city-is-capital", entity_family="ef-ireland",
         entities=["Ireland"], provenance=dict(kind="authored")),
    dict(id="i-geo005", text="Paris is the capital of France.",
         domain="geo-civics", template_family="tf-x-is-capital-of-y", entity_family="ef-france",
         entities=["Paris", "France"], provenance=dict(kind="authored")),
    dict(id="i-geo006", text="Paris serves as the capital of France.",
         domain="geo-civics", template_family="tf-x-serves-as-capital-of-y", entity_family="ef-france",
         entities=["Paris", "France"], provenance=dict(kind="authored")),
    dict(id="i-geo007", text="Paris is not the capital of France.",
         domain="geo-civics", template_family="tf-x-is-capital-of-y", entity_family="ef-france",
         entities=["Paris", "France"],
         provenance=dict(kind="perturbed", parent_id="i-geo005", transformation="negation")),
    dict(id="i-geo008", text="Portugal lies to the west of Spain.",
         domain="geo-civics", template_family="tf-x-lies-west-of-y", entity_family="ef-iberia",
         entities=["Portugal", "Spain"], provenance=dict(kind="authored")),
    dict(id="i-geo009", text="Spain lies to the west of Portugal.",
         domain="geo-civics", template_family="tf-x-lies-west-of-y", entity_family="ef-iberia",
         entities=["Portugal", "Spain"],
         provenance=dict(kind="perturbed", parent_id="i-geo008", transformation="relation_swap")),
    dict(id="i-geo010", text="Bolivia is a landlocked country.",
         domain="geo-civics", template_family="tf-x-is-landlocked", entity_family="ef-bolivia",
         entities=["Bolivia"], provenance=dict(kind="authored")),
    dict(id="i-geo011", text="Bolivia has a long Atlantic coastline.",
         domain="geo-civics", template_family="tf-x-has-coastline", entity_family="ef-bolivia",
         entities=["Bolivia"], provenance=dict(kind="authored")),
    dict(id="i-geo012", text="As of the 1990 census, the city had about 240,000 residents.",
         domain="geo-civics", template_family="tf-city-population-at-time", entity_family="ef-rivertown",
         entities=["Rivertown"], time_index="1990", provenance=dict(kind="authored")),
    dict(id="i-geo013", text="The city's population reached roughly 610,000 in the 2020 count.",
         domain="geo-civics", template_family="tf-city-population-at-time", entity_family="ef-rivertown",
         entities=["Rivertown"], time_index="2020", provenance=dict(kind="authored")),
    dict(id="i-geo014", text="The national parliament has two chambers.",
         domain="geo-civics", template_family="tf-parliament-structure", entity_family="ef-lindenia",
         entities=["Lindenia"], provenance=dict(kind="authored")),
    dict(id="i-geo015", text="The lower chamber of the parliament has 300 seats.",
         domain="geo-civics", template_family="tf-parliament-seats", entity_family="ef-lindenia",
         entities=["Lindenia"], provenance=dict(kind="authored")),
    dict(id="i-geo016", text="Cork is the capital of Ireland.",
         domain="geo-civics", template_family="tf-x-is-capital-of-y", entity_family="ef-ireland",
         entities=["Cork", "Ireland"], provenance=dict(kind="authored")),
    dict(id="i-geo017", text="The 1957 treaty created a customs union and abolished internal tariffs between members.",
         domain="geo-civics", template_family="tf-treaty-multi-provision", entity_family="ef-union-treaty",
         entities=["1957 treaty"], time_index="1957",
         atomic_claims=["the 1957 treaty created a customs union",
                        "the 1957 treaty abolished internal tariffs between members"],
         provenance=dict(kind="authored")),
    dict(id="i-geo018", text="Under the 1957 treaty, tariffs on goods traded between member states were removed.",
         domain="geo-civics", template_family="tf-treaty-single-provision", entity_family="ef-union-treaty",
         entities=["1957 treaty"], provenance=dict(kind="authored")),
    dict(id="i-geo019", text="The 1957 treaty was signed by six founding member states.",
         domain="geo-civics", template_family="tf-treaty-signatories", entity_family="ef-union-treaty",
         entities=["1957 treaty"], provenance=dict(kind="authored")),

    # --- everyday-statements -------------------------------------------------- #
    dict(id="i-eve001", text="The committee rejected the proposal.",
         domain="everyday-statements", template_family="tf-body-rejected-thing", entity_family="ef-committee-proposal",
         entities=["committee"], provenance=dict(kind="authored")),
    dict(id="i-eve002", text="The proposal was turned down by the committee.",
         domain="everyday-statements", template_family="tf-thing-turned-down-by-body", entity_family="ef-committee-proposal",
         entities=["committee"], provenance=dict(kind="authored")),
    dict(id="i-eve003", text="The committee did not reject the proposal.",
         domain="everyday-statements", template_family="tf-body-rejected-thing", entity_family="ef-committee-proposal",
         entities=["committee"],
         provenance=dict(kind="perturbed", parent_id="i-eve001", transformation="negation")),
    dict(id="i-eve004", text="The committee rejected the proposal unanimously.",
         domain="everyday-statements", template_family="tf-body-rejected-thing-adverb", entity_family="ef-committee-proposal",
         entities=["committee"], provenance=dict(kind="authored")),
    dict(id="i-eve005", text="A golden retriever is asleep on the porch.",
         domain="everyday-statements", template_family="tf-animal-state-location", entity_family="ef-porch-dog",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve006", text="A dog is on the porch.",
         domain="everyday-statements", template_family="tf-animal-location", entity_family="ef-porch-dog",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve007", text="The kettle is boiling in the kitchen.",
         domain="everyday-statements", template_family="tf-object-state-location", entity_family="ef-kitchen-kettle",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve008", text="The recipe calls for two cups of flour.",
         domain="everyday-statements", template_family="tf-recipe-quantity", entity_family="ef-recipe-flour",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve009", text="My train was delayed by about forty minutes this morning.",
         domain="everyday-statements", template_family="tf-transit-delay", entity_family="ef-morning-train",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve010", text="The vote on the motion passed unanimously.",
         domain="everyday-statements", template_family="tf-vote-outcome", entity_family="ef-motion-vote",
         entities=[], provenance=dict(kind="authored")),
    dict(id="i-eve011", text="Two members voted against the motion.",
         domain="everyday-statements", template_family="tf-vote-dissent", entity_family="ef-motion-vote",
         entities=[], provenance=dict(kind="authored")),

    # --- biomed-claims ------------------------------------------------------ #
    dict(id="i-bio001", text="The drug lowered blood pressure and improved sleep quality in elderly patients.",
         domain="biomed-claims", template_family="tf-drug-multi-effect-population", entity_family="ef-drug-x-elderly",
         entities=["drug X"],
         atomic_claims=[
             "the drug lowered blood pressure in elderly patients",
             "the drug improved sleep quality in elderly patients",
         ],
         provenance=dict(kind="authored")),
    dict(id="i-bio002", text="In a trial of elderly patients, the drug produced a significant reduction in blood pressure.",
         domain="biomed-claims", template_family="tf-trial-single-effect", entity_family="ef-drug-x-elderly",
         entities=["drug X"], provenance=dict(kind="authored")),
    dict(id="i-bio003", text="The drug did not lower blood pressure or improve sleep quality in elderly patients.",
         domain="biomed-claims", template_family="tf-drug-multi-effect-population", entity_family="ef-drug-x-elderly",
         entities=["drug X"],
         provenance=dict(kind="perturbed", parent_id="i-bio001", transformation="negation")),
    dict(id="i-bio004", text="The 2011 study reported a 12% relative risk reduction in the treatment arm.",
         domain="biomed-claims", template_family="tf-study-effect-at-time", entity_family="ef-cohort-trial",
         entities=["treatment arm"], time_index="2011", provenance=dict(kind="authored")),
    dict(id="i-bio005", text="A 2019 re-analysis of the same cohort found no significant risk reduction.",
         domain="biomed-claims", template_family="tf-study-effect-at-time", entity_family="ef-cohort-trial",
         entities=["treatment arm"], time_index="2019", provenance=dict(kind="authored")),
    dict(id="i-bio006", text="The compound reduced tumour growth in a mouse model.",
         domain="biomed-claims", template_family="tf-compound-effect-model", entity_family="ef-compound-y",
         entities=["compound Y"], provenance=dict(kind="authored")),
    dict(id="i-bio007", text="The compound was evaluated in a mouse model of cancer.",
         domain="biomed-claims", template_family="tf-compound-evaluated-model", entity_family="ef-compound-y",
         entities=["compound Y"], provenance=dict(kind="authored")),

    # --- corporate-events -------------------------------------------------- #
    dict(id="i-cor001", text="Acme Corp acquired Beta Systems in a cash deal.",
         domain="corporate-events", template_family="tf-x-acquired-y", entity_family="ef-acme-beta",
         entities=["Acme Corp", "Beta Systems"], provenance=dict(kind="authored")),
    dict(id="i-cor002", text="Beta Systems acquired Acme Corp in a cash deal.",
         domain="corporate-events", template_family="tf-x-acquired-y", entity_family="ef-acme-beta",
         entities=["Acme Corp", "Beta Systems"],
         provenance=dict(kind="perturbed", parent_id="i-cor001", transformation="relation_swap")),
    dict(id="i-cor003", text="In a cash transaction, Acme Corp took over Beta Systems.",
         domain="corporate-events", template_family="tf-x-took-over-y", entity_family="ef-acme-beta",
         entities=["Acme Corp", "Beta Systems"], provenance=dict(kind="authored")),
    dict(id="i-cor004", text="Acme Corp did not acquire Beta Systems in a cash deal.",
         domain="corporate-events", template_family="tf-x-acquired-y", entity_family="ef-acme-beta",
         entities=["Acme Corp", "Beta Systems"],
         provenance=dict(kind="perturbed", parent_id="i-cor001", transformation="negation")),
    dict(id="i-cor005", text="Acme Corp announced a new chief executive.",
         domain="corporate-events", template_family="tf-company-ceo-announcement", entity_family="ef-acme-beta",
         entities=["Acme Corp"], provenance=dict(kind="authored")),
    dict(id="i-cor006", text="At the end of the 2015 fiscal year, the firm employed 1,200 staff.",
         domain="corporate-events", template_family="tf-firm-headcount-at-time", entity_family="ef-northwind",
         entities=["Northwind"], time_index="2015", provenance=dict(kind="authored")),
    dict(id="i-cor007", text="By the 2024 annual report, the firm's headcount had grown to 8,500.",
         domain="corporate-events", template_family="tf-firm-headcount-at-time", entity_family="ef-northwind",
         entities=["Northwind"], time_index="2024", provenance=dict(kind="authored")),
    dict(id="i-cor008", text="The board approved the merger at its March meeting.",
         domain="corporate-events", template_family="tf-board-approved-thing", entity_family="ef-merger-vote",
         entities=["board"], provenance=dict(kind="authored")),
    dict(id="i-cor009", text="The merger was approved by the board at its March meeting.",
         domain="corporate-events", template_family="tf-thing-approved-by-board", entity_family="ef-merger-vote",
         entities=["board"], provenance=dict(kind="authored")),

    # --- product-support -------------------------------------------------- #
    # product-support: the product itself ("the printer" / "the app") is treated as
    # the named entity for this domain, so entity-related pairs have a real anchor.
    dict(id="i-pro001", text="my printer keeps jamming on thick card stock",
         domain="product-support", template_family="tf-device-fault-condition", entity_family="ef-printer-jam",
         entities=["the printer"], provenance=dict(kind="authored")),
    dict(id="i-pro002", text="the printer jams every time I load heavy card",
         domain="product-support", template_family="tf-device-fault-condition", entity_family="ef-printer-jam",
         entities=["the printer"], provenance=dict(kind="authored")),
    dict(id="i-pro003", text="the printer does not jam on thick card stock anymore after the firmware update",
         domain="product-support", template_family="tf-device-fault-resolved", entity_family="ef-printer-jam",
         entities=["the printer"], provenance=dict(kind="authored")),
    dict(id="i-pro004", text="the app crashes on launch after the 4.2 update",
         domain="product-support", template_family="tf-app-crash-after-version", entity_family="ef-app-crash",
         entities=["the app"], provenance=dict(kind="authored")),
    dict(id="i-pro005", text="since updating to 4.2 the app closes immediately when I open it",
         domain="product-support", template_family="tf-app-crash-after-version", entity_family="ef-app-crash",
         entities=["the app"], provenance=dict(kind="authored")),
    dict(id="i-pro006", text="how do I export my data to CSV from the reports screen in the app",
         domain="product-support", template_family="tf-howto-export", entity_family="ef-export-csv",
         entities=["the app"], provenance=dict(kind="authored")),
    dict(id="i-pro007", text="the app's reports screen has a CSV export button in the top right",
         domain="product-support", template_family="tf-feature-location", entity_family="ef-export-csv",
         entities=["the app"], provenance=dict(kind="authored")),

    # --- a couple of cross-domain fillers for 'unrelated' ----------------- #
    dict(id="i-eve012", text="It rained for most of the weekend.",
         domain="everyday-statements", template_family="tf-weather-report", entity_family="ef-weekend-weather",
         entities=[], provenance=dict(kind="authored")),
]

# --------------------------------------------------------------------------- #
# PAIRS  (build_dev.py computes lexical_overlap / char_3gram_overlap / entity_overlap and the pair id)
# --------------------------------------------------------------------------- #
# fields: a, b, relation, runner_up, [direction], [notes]
PAIRS = [
    # equivalent
    dict(a="i-geo001", b="i-geo002", relation="equivalent", runner_up="paraphrase",
         notes="same proposition, word order only"),
    dict(a="i-cor008", b="i-cor009", relation="equivalent", runner_up="paraphrase",
         notes="active/passive of the same event, no nuance change"),

    # paraphrase
    dict(a="i-eve001", b="i-eve002", relation="paraphrase", runner_up="equivalent",
         notes="turned down vs rejected — near-equivalent, kept as paraphrase for register"),
    dict(a="i-geo005", b="i-geo006", relation="paraphrase", runner_up="equivalent",
         notes="'serves as' adds a shade of role framing"),
    dict(a="i-cor001", b="i-cor003", relation="paraphrase", runner_up="equivalent",
         notes="acquired vs took over; both cash deal"),
    dict(a="i-pro001", b="i-pro002", relation="paraphrase", runner_up="equivalent",
         notes="user-written, low lexical overlap, same complaint"),

    # entailment (a -> b)
    dict(a="i-eve005", b="i-eve006", relation="entailment", runner_up="paraphrase", direction="a->b",
         notes="golden retriever asleep on porch -> a dog is on the porch; reverse fails"),
    dict(a="i-eve004", b="i-eve001", relation="entailment", runner_up="paraphrase", direction="a->b",
         notes="rejected unanimously -> rejected; reverse fails"),
    dict(a="i-bio006", b="i-bio007", relation="entailment", runner_up="topic-related", direction="a->b",
         notes="reduced tumour growth in a mouse model -> was evaluated in a mouse model"),

    # partial-support (b partially supports a)
    dict(a="i-bio001", b="i-bio002", relation="partial-support", runner_up="entailment", direction="a->b",
         notes="b supports the blood-pressure claim, silent on sleep quality"),
    dict(a="i-geo017", b="i-geo018", relation="partial-support", runner_up="entailment", direction="a->b",
         notes="b supports the tariff-abolition claim, silent on the customs union"),

    # contradiction (atemporal)
    dict(a="i-geo010", b="i-geo011", relation="contradiction", runner_up="topic-related",
         notes="landlocked vs long Atlantic coastline; no time index -> contradiction not temporal-mismatch"),
    dict(a="i-eve010", b="i-eve011", relation="contradiction", runner_up="topic-related",
         notes="passed unanimously vs two voted against; low lexical overlap contradiction"),
    dict(a="i-geo001", b="i-geo016", relation="contradiction", runner_up="topic-related",
         notes="Dublin vs Cork as capital of Ireland; high lexical overlap, atemporal incompatibility"),

    # negation (explicit not, overlap >= 0.7)
    dict(a="i-geo001", b="i-geo003", relation="negation", runner_up="contradiction", direction="a->b",
         notes="minimal edit: insert 'not'"),
    dict(a="i-geo005", b="i-geo007", relation="negation", runner_up="contradiction", direction="a->b"),
    dict(a="i-eve001", b="i-eve003", relation="negation", runner_up="contradiction", direction="a->b"),
    dict(a="i-bio001", b="i-bio003", relation="negation", runner_up="contradiction", direction="a->b",
         notes="negates the first atomic claim; drops the second — still a valid negation of the main predicate"),
    dict(a="i-cor001", b="i-cor004", relation="negation", runner_up="contradiction", direction="a->b"),

    # temporal-mismatch (both items have time_index)
    dict(a="i-geo012", b="i-geo013", relation="temporal-mismatch", runner_up="contradiction", direction="a->b",
         notes="same city population, 1990 vs 2020"),
    dict(a="i-bio004", b="i-bio005", relation="temporal-mismatch", runner_up="contradiction", direction="a->b",
         notes="2011 effect vs 2019 re-analysis of the same cohort"),
    dict(a="i-cor006", b="i-cor007", relation="temporal-mismatch", runner_up="contradiction", direction="a->b",
         notes="firm headcount 2015 vs 2024"),

    # relation-swap (shared entities, roles reversed)
    dict(a="i-cor001", b="i-cor002", relation="relation-swap", runner_up="contradiction", direction="a->b",
         notes="Acme acquired Beta -> Beta acquired Acme"),
    dict(a="i-geo008", b="i-geo009", relation="relation-swap", runner_up="contradiction", direction="a->b",
         notes="Portugal west of Spain -> Spain west of Portugal"),

    # topic-related (same topic, independent claims)
    dict(a="i-geo014", b="i-geo015", relation="topic-related", runner_up="entailment",
         notes="two chambers vs lower chamber seat count — actually 015 entails a chamber; keep as topic-related test for adjudication"),
    dict(a="i-bio004", b="i-bio006", relation="topic-related", runner_up="entity-related",
         notes="both about drug trials/effects, different compounds and claims"),
    dict(a="i-cor005", b="i-cor008", relation="topic-related", runner_up="entity-related",
         notes="both corporate-governance events at (different) firms"),
    dict(a="i-geo017", b="i-geo019", relation="topic-related", runner_up="entity-related",
         notes="both about the 1957 treaty; provisions vs signatories - independent claims, mid/high overlap"),
    dict(a="i-geo018", b="i-geo019", relation="topic-related", runner_up="entity-related",
         notes="both about the 1957 treaty; tariff removal vs signatories"),

    # entity-related (shared entity, different topic)
    dict(a="i-cor001", b="i-cor005", relation="entity-related", runner_up="topic-related",
         notes="both about Acme Corp; acquisition vs CEO announcement"),
    dict(a="i-pro004", b="i-pro007", relation="entity-related", runner_up="topic-related",
         notes="both about the app; crash bug vs feature location - different sub-topics"),
    dict(a="i-pro004", b="i-pro006", relation="entity-related", runner_up="topic-related",
         notes="both about the app; a crash bug vs a how-to question - shared entity, different topics"),

    # unrelated (cross-domain, ~0 overlap, no shared entity)
    dict(a="i-geo001", b="i-pro001", relation="unrelated", runner_up=None,
         notes="capital of Ireland vs printer jam"),
    dict(a="i-bio006", b="i-eve012", relation="unrelated", runner_up=None,
         notes="tumour growth in mice vs weekend weather"),
    dict(a="i-cor006", b="i-eve008", relation="unrelated", runner_up=None,
         notes="firm headcount vs a recipe"),
    dict(a="i-pro006", b="i-eve009", relation="unrelated", runner_up=None,
         notes="CSV export how-to vs a delayed train"),
]

# --------------------------------------------------------------------------- #
# QUERIES  (build_dev.py assigns ids)
# --------------------------------------------------------------------------- #
# fields: text, query_style, target, positives[(item,grade)], hard_negatives[(item,underlying,method)]
QUERIES = [
    dict(text="Is Dublin the capital of Ireland?", query_style="question", target="i-geo001",
         positives=[("i-geo001", 3), ("i-geo002", 3), ("i-geo004", 2)],
         hard_negatives=[("i-geo003", "negation", "structured_perturbation"),
                         ("i-geo005", "entity-related", "entity_matched")]),
    dict(text="Which company acquired Beta Systems?", query_style="question", target="i-cor001",
         positives=[("i-cor001", 3), ("i-cor003", 3)],
         hard_negatives=[("i-cor002", "relation-swap", "structured_perturbation"),
                         ("i-cor004", "negation", "structured_perturbation")]),
    dict(text="did the drug lower blood pressure in elderly patients", query_style="keyword", target="i-bio001",
         positives=[("i-bio001", 3), ("i-bio002", 2)],
         hard_negatives=[("i-bio003", "negation", "structured_perturbation"),
                         ("i-bio005", "temporal-mismatch", "structured_perturbation")]),
]
