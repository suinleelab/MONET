def concept_to_prompt(concept_name):
    if concept_name == "White(Hypopigmentation)":
        text_counter = "White"
        prompt_dict = {"original": "This is White(Hypopigmentation)"}
        prompt_dict.update(
            {
                "1": ["This is White(Hypopigmentation)"],
                "1_": ["This photo is White(Hypopigmentation)"],
                "2": ["This is Hypopigmentation"],
                "2_": ["This photo is Hypopigmentation"],
            }
        )
        prompt_dict["original"] = "This is White(Hypopigmentation)"

    elif concept_name == "Brown(Hyperpigmentation)":
        text_counter = "Brown"
        prompt_dict = {"original": "This is Brown(Hyperpigmentation)"}
        prompt_dict.update(
            {
                "1": ["This is Brown(Hyperpigmentation)"],
                "1_": ["This photo is Brown(Hyperpigmentation)"],
                "2": ["This is Hyperpigmentation"],
                "2_": ["This photo is Hyperpigmentation"],
                "3": ["This is Hyperpigmented"],
                "3_": ["This photo is Hyperpigmented"],
            }
        )

    elif concept_name == "Blue":
        text_counter = "Blue"
        prompt_dict = {"original": "This is Blue"}
        prompt_dict.update(
            {
                "1": ["This is Blue"],
                "2": ["This lesion is Blue"],
                "3": ["This lesion is Blue color"],
            }
        )

    elif concept_name == "Yellow":
        text_counter = "Yellow"
        prompt_dict = {"original": "This is Yellow"}
        prompt_dict.update(
            {
                "1": ["This is Yellow"],
            }
        )

    elif concept_name == "Black":
        text_counter = "Black"
        prompt_dict = {"original": "This is Black"}
        prompt_dict.update(
            {
                "1": ["This is Black"],
                "2": ["This lesion is Black"],
                "3": ["This lesion is Black color"],
            }
        )

    elif concept_name == "Purple":
        text_counter = "Purple"
        prompt_dict = {"original": "This is Purple"}
        prompt_dict.update(
            {
                "1": ["This is Purple"],
            }
        )

    elif concept_name == "Gray":
        text_counter = "Gray"
        prompt_dict = {"original": "This is Gray"}
        prompt_dict.update(
            {
                "1": ["This photo is Gray"],
            }
        )

    elif concept_name == "Pigmented":
        text_counter = "Pigmented"
        prompt_dict = {"original": "This is Pigmented"}
        prompt_dict.update(
            {
                "1": ["This is Pigmented"],
            }
        )

    elif concept_name == "Erythema":
        text_counter = "Erythema"
        prompt_dict = {"original": "This is Erythema"}
        prompt_dict.update(
            {
                "1": ["This is redness"],
                "2": ["This is erythematous"],
            }
        )

    ################################
    # primary
    ################################
    elif concept_name == "Patch":
        text_counter = "Patch"
        prompt_dict = {"original": "This is Patch"}
        prompt_dict.update(
            {
                "1": ["This is Vitiligo"],
                "1_": ["This photo is Vitiligo"],
                "2": ["This is Melasma"],
                "2_": ["This photo is Melasma"],
                "3": ["This is hyperpigmented"],
                "3_": ["This is hyperpigmented"],
            }
        )

    elif concept_name == "Nodule":
        text_counter = "Nodul"
        prompt_dict = {"original": "This is Nodule"}
        prompt_dict.update(
            {
                "1": ["This is Nodule"],
                "2": ["This is nodular"],
                "3": ["This is cyst"],
            }
        )

    elif concept_name == "Macule":
        text_counter = "Macul"
        prompt_dict = {"original": "This is Macule"}
        prompt_dict.update(
            {
                "1": ["This is Macular"],
                "2": ["This photo is Macule"],
                "3": ["This is Lentigo"],
                "4": ["This photo is Lentigo"],
                "5": ["This is freckle"],
                "6": ["This photo is freckle"],
            }
        )

    elif concept_name == "Papule":
        text_counter = "Papul"
        prompt_dict = {"original": "This is Papule"}
        prompt_dict.update(
            {
                "1": ["This is Papular"],
            }
        )

    elif concept_name == "Plaque":
        text_counter = "Plaqu"
        prompt_dict = {"original": "This is Plaque"}
        prompt_dict.update(
            {
                "1": ["This is Plaque"],
                "2": ["This is Psoriasis"],
                "3": ["This is dermatitis"],
            }
        )

    elif concept_name == "Vesicle":
        text_counter = "Vesicl"
        prompt_dict = {"original": "This is Vesicle"}
        prompt_dict.update(
            {
                "1": ["This photo is Vesicle"],
                "2": ["This is fluid-containing"],
            }
        )

    elif concept_name == "Pustule":
        text_counter = "Pustul"
        prompt_dict = {"original": "This is Pustule"}
        prompt_dict.update(
            {
                "1": ["This photo is Pustule"],
            }
        )

    ################################
    # secondary
    ################################

    elif concept_name == "Crust":
        text_counter = "Crust"
        prompt_dict = {"original": "This is Crust"}
        prompt_dict.update(
            {
                "1": ["This is Crust"],
                "2": ["This is dried Crust"],
                "2_": ["This photo is dried Crust"],
            }
        )

    elif concept_name == "Scale":
        text_counter = "Scale"
        prompt_dict = {"original": "This is Scale"}
        prompt_dict.update(
            {
                "1": ["Hyperkeratosis"],
                "2": ["This is scaly"],
                # "3": ["This is flaking scale"],
                "3": ["This is flaky and scaly"],
            }
        )
    elif concept_name == "Fissure":
        text_counter = "Fissure"
        prompt_dict = {"original": "This is Fissure"}
        prompt_dict.update(
            {
                "1": ["This is dry and cracked skin"],
            }
        )

    elif concept_name == "Erosion":
        text_counter = "Erosion"
        prompt_dict = {"original": "This is Erosion"}
        prompt_dict.update(
            {
                "1": ["This is Erosion"],
                "2": ["This photo is erosion"],
                "3": ["This is breakdown of the outer layers"],
                "4": ["This is Impetigo"],
                "5": ["This is Erosive"],
            }
        )

    elif concept_name == "Ulcer":
        text_counter = "Ulcer"
        prompt_dict = {"original": "This is Ulcer"}
        prompt_dict.update(
            {
                "1": ["This is Ulcer"],
                "2": ["This photo is Ulcer"],
                "3": ["This photo is Ulcerated"],
                "4": ["This is Ulcerated"],
            }
        )

    elif concept_name == "Excoriation":
        text_counter = "Excoriation"
        prompt_dict = {"original": "This is Excoriation"}
        prompt_dict.update(
            {
                "1": ["This photo is Excoriation"],
            }
        )

    elif concept_name == "Atrophy":
        text_counter = "Atrophy"
        prompt_dict = {"original": "This is Atrophy"}
        prompt_dict.update(
            {
                "1": ["This is Atrophic"],
            }
        )

    elif concept_name == "Lichenification":
        text_counter = "Lichenification"
        prompt_dict = {"original": "This is Lichenification"}
        prompt_dict.update(
            {
                "1": ["This is Lichenification"],
                "2": ["skin has become thickened and leathery"],
            }
        )

    ################################
    # others
    ################################

    elif concept_name == "Cyst":
        text_counter = "Cyst"
        prompt_dict = {"original": "This is Cyst"}
        prompt_dict.update(
            {
                "1": ["This photo is Cyst"],
            }
        )

    elif concept_name == "Salmon":
        text_counter = "Salmon"
        prompt_dict = {"original": "This is Salmon"}
        prompt_dict.update(
            {
                "1": ["This photo is Salmon patch"],
            }
        )

    elif concept_name == "Translucent":
        text_counter = "Translucent"
        prompt_dict = {"original": "This is Translucent"}
        prompt_dict.update(
            {
                "1": ["This is Translucent"],
                "2": ["This bump is Translucent"],
            }
        )

    elif concept_name == "Warty/Papillomatous":
        text_counter = "Wart"
        prompt_dict = {"original": "This is Warty/Papillomatous"}
        prompt_dict.update(
            {
                "1": ["This is Warty and Papillomatous"],
            }
        )
    elif concept_name == "Exophytic/Fungating":
        text_counter = "Exophyti"
        prompt_dict = {"original": "This is Exophytic/Fungating"}
        prompt_dict.update(
            {
                "1": ["This is Fungating"],
            }
        )

    elif concept_name == "Purpura/Petechiae":
        text_counter = "Purpura"
        prompt_dict = {"original": "This is Purpura/Petechiae"}
        prompt_dict.update(
            {
                "1": ["This is Purpura"],
            }
        )

    elif concept_name == "Friable":
        text_counter = "Friable"
        prompt_dict = {"original": "This is Friable"}
        prompt_dict.update(
            {
                "1": ["This photo is Friable"],
                "2": ["This is Friable"],
            }
        )

    elif concept_name == "Bulla":
        text_counter = "bullae"
        prompt_dict = {"original": "This is Bulla"}
        prompt_dict.update(
            {
                "1": ["This photo is bullae"],
                "2": ["This is bullae"],
                "3": ["This is blister"],
                "4": ["This photo is blister"],
            }
        )

    elif concept_name == "Xerosis":
        text_counter = "Xerosis"
        prompt_dict = {"original": "This is Xerosis"}
        prompt_dict.update(
            {
                "1": ["This photo is Xerosis"],
                "2": ["This is Xerosis"],
                "3": ["This is abnormally dry skin"],
                "4": ["This photo is abnormally dry skin"],
                "5": ["This is dry skin"],
                "6": ["This photo is dry skin"],
            }
        )

    elif concept_name == "Scar":
        text_counter = "Scar"
        prompt_dict = {"original": "This is Scar"}
        prompt_dict.update(
            {
                "1": ["This photo is Scar"],
                "2": ["This is Scar"],
                "3": ["This is Keloid scars"],
                "4": ["This is Contractures scars"],
                "5": ["This is Hypertrophic scars"],
                "6": ["This is Acnescars scars"],
            }
        )
    elif concept_name == "Sclerosis":
        text_counter = "Sclerosis"
        prompt_dict = {"original": "This is Sclerosis"}
        prompt_dict.update(
            {
                "1": ["This is Scleroderma"],
                "2": ["This is CREST syndrome"],
            }
        )

    elif concept_name == "Abscess":
        text_counter = "Abscess"
        prompt_dict = {"original": "This is Abscess"}
        prompt_dict.update(
            {
                "1": ["This is Abscess"],
                "2": ["This is swollen, pus-filled lump"],
            }
        )

    elif concept_name == "Exudate":
        text_counter = "Exudate"
        prompt_dict = {"original": "This is Exudate"}
        prompt_dict.update(
            {
                "1": ["This is Exudate"],
                "2": ["This is Ooze. Pus. Secretion"],
            }
        )

    elif concept_name == "Acuminate":  # THIS DOES NOT WORK WELL
        text_counter = "Acuminate"
        prompt_dict = {"original": "This is Acuminate"}
        prompt_dict.update(
            {
                "1": ["This is Acuminate"],
            }
        )

    elif concept_name == "Burrow":
        text_counter = "Burrow"
        prompt_dict = {"original": "This is Burrow"}
        prompt_dict.update(
            {
                "1": ["This is Burrow"],
                "2": ["This photo is Burrow"],
                "3": ["This is Scabies"],
                "4": ["This photo is Scabies"],
            }
        )

    elif concept_name == "Wheal":
        text_counter = "Urticaria"
        prompt_dict = {"original": "This is Wheal"}
        prompt_dict.update(
            {
                "1": ["This is Urticaria"],
                "2": ["This photo is Urticaria"],
            }
        )

    elif concept_name == "Comedo":  # ISN'T IT DISEASE?
        text_counter = "Comedo"
        prompt_dict = {"original": "This is Comedo"}
        prompt_dict.update(
            {
                "1": ["This photo is whitehead or blackhead"],
            }
        )

    elif concept_name == "Induration":
        text_counter = "Induration"
        prompt_dict = {"original": "This is Induration"}
        prompt_dict.update(
            {
                "1": ["This is Edema"],
                "2": ["This is oedema"],
            }
        )

    elif concept_name == "Telangiectasia":
        text_counter = "Telangiectasia"
        prompt_dict = {"original": "This is Telangiectasia"}
        prompt_dict.update(
            {
                "1": ["This is dilated or broken blood vessels"],
                "2": ["This photo is dilated or broken blood vessels"],
            }
        )

    elif concept_name == "Pedunculated":
        text_counter = "Pedunculated"
        prompt_dict = {"original": "This is Pedunculated"}
        prompt_dict.update(
            {
                "1": ["This is Pedunculated"],
                "2": ["This photo is Pedunculated"],
            }
        )

    elif concept_name == "Poikiloderma":
        text_counter = "Poikiloderma"
        prompt_dict = {"original": "This is Poikiloderma"}
        prompt_dict.update(
            {
                "1": ["This is sun aging"],
                "2": ["This photo is sun aging"],
            }
        )

    elif concept_name == "Umbilicated":
        text_counter = "Umbilicated"
        prompt_dict = {"original": "This is Umbilicated"}
        prompt_dict.update(
            {
                "1": ["This is Umbilicated"],
            }
        )

    elif concept_name == "Dome-shaped":
        text_counter = "Dome"
        prompt_dict = {"original": "This is Dome-shaped"}
        prompt_dict.update(
            {
                "1": ["This is like Dome"],
            }
        )

    elif concept_name == "Flat topped":
        text_counter = "Flat"
        prompt_dict = {"original": "This is Flat topped"}
        prompt_dict.update(
            {
                "1": ["This is Flat topped"],
            }
        )
    else:
        raise ValueError(f"Concept {concept_name} not found")
    return prompt_dict, text_counter


skincon_cols = [
    "skincon_Vesicle",
    "skincon_Papule",
    "skincon_Macule",
    "skincon_Plaque",
    "skincon_Abscess",
    "skincon_Pustule",
    "skincon_Bulla",
    "skincon_Patch",
    "skincon_Nodule",
    "skincon_Ulcer",
    "skincon_Crust",
    "skincon_Erosion",
    "skincon_Excoriation",
    "skincon_Atrophy",
    "skincon_Exudate",
    "skincon_Purpura/Petechiae",
    "skincon_Fissure",
    "skincon_Induration",
    "skincon_Xerosis",
    "skincon_Telangiectasia",
    "skincon_Scale",
    "skincon_Scar",
    "skincon_Friable",
    "skincon_Sclerosis",
    "skincon_Pedunculated",
    "skincon_Exophytic/Fungating",
    "skincon_Warty/Papillomatous",
    "skincon_Dome-shaped",
    "skincon_Flat topped",
    "skincon_Brown(Hyperpigmentation)",
    "skincon_Translucent",
    "skincon_White(Hypopigmentation)",
    "skincon_Purple",
    "skincon_Yellow",
    "skincon_Black",
    "skincon_Erythema",
    "skincon_Comedo",
    "skincon_Lichenification",
    "skincon_Blue",
    "skincon_Umbilicated",
    "skincon_Poikiloderma",
    "skincon_Salmon",
    "skincon_Wheal",
    "skincon_Acuminate",
    "skincon_Burrow",
    "skincon_Gray",
    "skincon_Pigmented",
    "skincon_Cyst",
]

fitzpatrick17k_threelabel = ["non-neoplastic", "benign", "malignant"]

fitzpatrick17k_disease_label = [
    "drug induced pigmentary changes",
    "photodermatoses",
    "dermatofibroma",
    "psoriasis",
    "kaposi sarcoma",
    "neutrophilic dermatoses",
    "granuloma annulare",
    "nematode infection",
    "allergic contact dermatitis",
    "necrobiosis lipoidica",
    "hidradenitis",
    "melanoma",
    "acne vulgaris",
    "sarcoidosis",
    "xeroderma pigmentosum",
    "actinic keratosis",
    "scleroderma",
    "syringoma",
    "folliculitis",
    "pityriasis lichenoides chronica",
    "porphyria",
    "dyshidrotic eczema",
    "seborrheic dermatitis",
    "prurigo nodularis",
    "acne",
    "neurofibromatosis",
    "eczema",
    "pediculosis lids",
    "basal cell carcinoma",
    "pityriasis rubra pilaris",
    "pityriasis rosea",
    "livedo reticularis",
    "stevens johnson syndrome",
    "erythema multiforme",
    "acrodermatitis enteropathica",
    "epidermolysis bullosa",
    "dermatomyositis",
    "urticaria",
    "basal cell carcinoma morpheiform",
    "vitiligo",
    "erythema nodosum",
    "lupus erythematosus",
    "lichen planus",
    "sun damaged skin",
    "drug eruption",
    "scabies",
    "cheilitis",
    "urticaria pigmentosa",
    "behcets disease",
    "nevocytic nevus",
    "mycosis fungoides",
    "superficial spreading melanoma ssm",
    "porokeratosis of mibelli",
    "juvenile xanthogranuloma",
    "milia",
    "granuloma pyogenic",
    "papilomatosis confluentes and reticulate",
    "neurotic excoriations",
    "epidermal nevus",
    "naevus comedonicus",
    "erythema annulare centrifigum",
    "pilar cyst",
    "pustular psoriasis",
    "ichthyosis vulgaris",
    "lyme disease",
    "striae",
    "rhinophyma",
    "calcinosis cutis",
    "stasis edema",
    "neurodermatitis",
    "congenital nevus",
    "squamous cell carcinoma",
    "mucinosis",
    "keratosis pilaris",
    "keloid",
    "tuberous sclerosis",
    "acquired autoimmune bullous diseaseherpes gestationis",
    "fixed eruptions",
    "lentigo maligna",
    "lichen simplex",
    "dariers disease",
    "lymphangioma",
    "pilomatricoma",
    "lupus subacute",
    "perioral dermatitis",
    "disseminated actinic porokeratosis",
    "erythema elevatum diutinum",
    "halo nevus",
    "aplasia cutis",
    "incontinentia pigmenti",
    "tick bite",
    "fordyce spots",
    "telangiectases",
    "solid cystic basal cell carcinoma",
    "paronychia",
    "becker nevus",
    "pyogenic granuloma",
    "langerhans cell histiocytosis",
    "port wine stain",
    "malignant melanoma",
    "factitial dermatitis",
    "xanthomas",
    "nevus sebaceous of jadassohn",
    "hailey hailey disease",
    "scleromyxedema",
    "porokeratosis actinic",
    "rosacea",
    "acanthosis nigricans",
    "myiasis",
    "seborrheic keratosis",
    "mucous cyst",
    "lichen amyloidosis",
    "ehlers danlos syndrome",
    "tungiasis",
]


fitzpatrick17k_ninelabel = [
    "inflammatory",
    "benign dermal",
    "malignant dermal",
    "malignant melanoma",
    "genodermatoses",
    "malignant epidermal",
    "benign epidermal",
    "benign melanocyte",
    "malignant cutaneous lymphoma",
]

ddi_map = {
    "acral-melanotic-macule": "melanoma look-alike",
    "atypical-spindle-cell-nevus-of-reed": "melanoma look-alike",
    "benign-keratosis": "melanoma look-alike",
    "blue-nevus": "melanoma look-alike",
    "dermatofibroma": "melanoma look-alike",
    "dysplastic-nevus": "melanoma look-alike",
    "epidermal-nevus": "melanoma look-alike",
    "hyperpigmentation": "melanoma look-alike",
    "keloid": "melanoma look-alike",
    "inverted-follicular-keratosis": "melanoma look-alike",
    "melanocytic-nevi": "melanoma look-alike",
    "melanoma": "melanoma",
    "melanoma-acral-lentiginous": "melanoma",
    "melanoma-in-situ": "melanoma",
    "nevus-lipomatosus-superficialis": "melanoma look-alike",
    "nodular-melanoma-(nm)": "melanoma",
    "pigmented-spindle-cell-nevus-of-reed": "melanoma look-alike",
    "seborrheic-keratosis": "melanoma look-alike",
    "seborrheic-keratosis-irritated": "melanoma look-alike",
    "solar-lentigo": "melanoma look-alike",
}

ham10k_dx_labels = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]


derm7pt_diagnosis_labels = [
    "basal cell carcinoma",
    "blue nevus",
    "clark nevus",
    "combined nevus",
    "congenital nevus",
    "dermal nevus",
    "dermatofibroma",
    "lentigo",
    "melanoma (in situ)",
    "melanoma (less than 0.76 mm)",
    "melanoma (0.76 to 1.5 mm)",
    "melanoma (more than 1.5 mm)",
    "melanoma metastasis",
    "melanosis",
    "miscellaneous",
    "recurrent nevus",
    "reed or spitz nevus",
    "seborrheic keratosis",
    "vascular lesion",
    "melanoma",
]
