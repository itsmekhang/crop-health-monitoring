"""
Treatment recommendations and severity levels per disease class.
"""

RECOMMENDATIONS = {
    "Pepper__bell___Bacterial_spot": {
        "severity": "High",
        "description": "Bacterial infection causing dark, water-soaked spots on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide (e.g., Kocide) immediately",
            "Remove and destroy all infected plant material",
            "Avoid overhead irrigation — use drip irrigation instead",
            "Do not work with plants when wet to prevent spread",
        ],
        "prevention": "Use disease-free certified seeds. Rotate crops every 2-3 years.",
    },
    "Pepper__bell___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },
    "Potato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark brown spots with concentric rings on older leaves.",
        "treatment": [
            "Apply fungicide containing chlorothalonil or mancozeb",
            "Remove infected lower leaves to slow spread",
            "Ensure adequate plant spacing for airflow",
            "Water at the base of plants, not overhead",
        ],
        "prevention": "Plant certified disease-free seed potatoes. Maintain soil fertility.",
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "description": "Highly destructive fungal disease — can destroy an entire crop within days.",
        "treatment": [
            "Apply fungicide immediately (metalaxyl or cymoxanil)",
            "Destroy all infected plants — do not compost",
            "Harvest remaining healthy tubers early if infestation is severe",
            "Alert neighboring farms as this spreads rapidly",
        ],
        "prevention": "Use resistant varieties. Monitor weather — disease thrives in cool, wet conditions.",
    },
    "Potato___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },
    "Tomato_Bacterial_spot": {
        "severity": "High",
        "description": "Bacterial infection causing small, dark, water-soaked lesions on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide weekly",
            "Remove and bag all infected leaves and fruit",
            "Switch to drip irrigation to keep foliage dry",
            "Disinfect tools between plants",
        ],
        "prevention": "Use resistant varieties and certified transplants.",
    },
    "Tomato_Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark spots with concentric rings, starting on older leaves.",
        "treatment": [
            "Apply chlorothalonil or copper fungicide every 7-10 days",
            "Prune affected lower leaves",
            "Mulch around plants to prevent soil splash",
            "Improve air circulation by staking and pruning",
        ],
        "prevention": "Rotate crops. Avoid wetting foliage when watering.",
    },
    "Tomato_Late_blight": {
        "severity": "Critical",
        "description": "Rapidly spreading fungal disease — same pathogen that caused the Irish Potato Famine.",
        "treatment": [
            "Apply fungicide immediately (metalaxyl, cymoxanil, or chlorothalonil)",
            "Remove and destroy all infected plants",
            "Do not save seeds from infected plants",
            "Scout field daily and act fast — disease spreads within 24-48 hours",
        ],
        "prevention": "Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Tomato_Leaf_Mold": {
        "severity": "Moderate",
        "description": "Fungal disease causing yellow patches on upper leaf surface and olive-green mold below.",
        "treatment": [
            "Improve greenhouse/field ventilation",
            "Apply fungicide (chlorothalonil or copper-based)",
            "Reduce humidity — avoid overhead watering",
            "Remove heavily infected leaves",
        ],
        "prevention": "Use resistant varieties. Keep humidity below 85%.",
    },
    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing small circular spots with dark borders and light centers.",
        "treatment": [
            "Apply fungicide (chlorothalonil, mancozeb, or copper) every 7-14 days",
            "Remove infected leaves starting from the bottom of the plant",
            "Avoid working in wet fields",
            "Mulch soil to reduce spore splash",
        ],
        "prevention": "Rotate crops at least 2 years. Remove plant debris after harvest.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Pest infestation causing stippled, bronzed leaves. Mites visible under leaves.",
        "treatment": [
            "Spray plants with water to dislodge mites",
            "Apply miticide (abamectin or bifenazate)",
            "Introduce predatory mites (Phytoseiulus persimilis) as biological control",
            "Avoid broad-spectrum insecticides that kill natural predators",
        ],
        "prevention": "Monitor regularly. Spider mites thrive in hot, dry conditions — maintain adequate irrigation.",
    },
    "Tomato__Target_Spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing brown lesions with concentric rings resembling a target.",
        "treatment": [
            "Apply fungicide (azoxystrobin or chlorothalonil)",
            "Remove infected plant debris",
            "Improve air circulation",
            "Avoid overhead irrigation",
        ],
        "prevention": "Use resistant varieties. Practice crop rotation.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "Critical",
        "description": "Viral disease spread by whiteflies causing severe leaf curling and yellowing. No cure.",
        "treatment": [
            "Remove and destroy infected plants immediately — there is no cure",
            "Control whitefly populations with insecticide (imidacloprid)",
            "Use yellow sticky traps to monitor whitefly levels",
            "Install insect-proof netting in greenhouses",
        ],
        "prevention": "Plant virus-resistant varieties. Control whiteflies before planting.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "severity": "High",
        "description": "Viral disease causing mottled, mosaic-patterned leaves and stunted growth. No cure.",
        "treatment": [
            "Remove and destroy infected plants — there is no cure",
            "Disinfect hands and tools with bleach solution after handling infected plants",
            "Control aphid vectors with insecticide",
            "Do not smoke near plants (tobacco mosaic virus can spread from cigarettes)",
        ],
        "prevention": "Use certified virus-free seeds. Wash hands before handling plants.",
    },
    "Tomato_healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # ── New triple-underscore aliases (full 39-class model) ────────────────
    # Tomato
    "Tomato___Bacterial_spot": {
        "severity": "High",
        "description": "Bacterial infection causing small, dark, water-soaked lesions on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide weekly",
            "Remove and bag all infected leaves and fruit",
            "Switch to drip irrigation to keep foliage dry",
            "Disinfect tools between plants",
        ],
        "prevention": "Use resistant varieties and certified transplants.",
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark spots with concentric rings, starting on older leaves.",
        "treatment": [
            "Apply chlorothalonil or copper fungicide every 7-10 days",
            "Prune affected lower leaves",
            "Mulch around plants to prevent soil splash",
            "Improve air circulation by staking and pruning",
        ],
        "prevention": "Rotate crops. Avoid wetting foliage when watering.",
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "description": "Rapidly spreading fungal disease — same pathogen that caused the Irish Potato Famine.",
        "treatment": [
            "Apply fungicide immediately (metalaxyl, cymoxanil, or chlorothalonil)",
            "Remove and destroy all infected plants",
            "Do not save seeds from infected plants",
            "Scout field daily — disease spreads within 24-48 hours",
        ],
        "prevention": "Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "description": "Fungal disease causing yellow patches on upper leaf surface and olive-green mold below.",
        "treatment": [
            "Improve ventilation",
            "Apply fungicide (chlorothalonil or copper-based)",
            "Reduce humidity — avoid overhead watering",
            "Remove heavily infected leaves",
        ],
        "prevention": "Use resistant varieties. Keep humidity below 85%.",
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing small circular spots with dark borders and light centers.",
        "treatment": [
            "Apply fungicide (chlorothalonil, mancozeb, or copper) every 7-14 days",
            "Remove infected leaves from the bottom up",
            "Avoid working in wet fields",
            "Mulch soil to reduce spore splash",
        ],
        "prevention": "Rotate crops at least 2 years. Remove plant debris after harvest.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Pest infestation causing stippled, bronzed leaves. Mites visible under leaves.",
        "treatment": [
            "Spray plants with water to dislodge mites",
            "Apply miticide (abamectin or bifenazate)",
            "Introduce predatory mites as biological control",
            "Avoid broad-spectrum insecticides that kill natural predators",
        ],
        "prevention": "Monitor regularly. Spider mites thrive in hot, dry conditions.",
    },
    "Tomato___Target_Spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing brown lesions with concentric rings resembling a target.",
        "treatment": [
            "Apply fungicide (azoxystrobin or chlorothalonil)",
            "Remove infected plant debris",
            "Improve air circulation",
            "Avoid overhead irrigation",
        ],
        "prevention": "Use resistant varieties. Practice crop rotation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Critical",
        "description": "Viral disease spread by whiteflies causing severe leaf curling and yellowing. No cure.",
        "treatment": [
            "Remove and destroy infected plants immediately — there is no cure",
            "Control whitefly populations with insecticide (imidacloprid)",
            "Use yellow sticky traps to monitor whitefly levels",
            "Install insect-proof netting in greenhouses",
        ],
        "prevention": "Plant virus-resistant varieties. Control whiteflies before planting.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "High",
        "description": "Viral disease causing mottled, mosaic-patterned leaves and stunted growth. No cure.",
        "treatment": [
            "Remove and destroy infected plants — there is no cure",
            "Disinfect hands and tools with bleach solution",
            "Control aphid vectors with insecticide",
        ],
        "prevention": "Use certified virus-free seeds. Wash hands before handling plants.",
    },
    "Tomato___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Pepper (new naming with comma)
    "Pepper,_bell___Bacterial_spot": {
        "severity": "High",
        "description": "Bacterial infection causing dark, water-soaked spots on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide (e.g., Kocide) immediately",
            "Remove and destroy all infected plant material",
            "Avoid overhead irrigation — use drip irrigation",
            "Do not work with plants when wet",
        ],
        "prevention": "Use disease-free certified seeds. Rotate crops every 2-3 years.",
    },
    "Pepper,_bell___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Apple
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
        "treatment": [
            "Apply fungicide (captan or myclobutanil) at bud break and every 7-10 days",
            "Remove and destroy fallen leaves to reduce spore load",
            "Prune for air circulation",
        ],
        "prevention": "Plant scab-resistant varieties. Rake and destroy fallen leaves each autumn.",
    },
    "Apple___Black_rot": {
        "severity": "High",
        "description": "Fungal disease causing concentric ring lesions on fruit and frogeye leaf spots.",
        "treatment": [
            "Apply fungicide (captan or thiophanate-methyl)",
            "Remove mummified fruit and dead wood — primary sources of infection",
            "Prune out cankers and destroy pruned material",
        ],
        "prevention": "Maintain good tree hygiene. Remove all mummified fruit before spring.",
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Fungal disease requiring two hosts (apple and cedar/juniper) to complete its lifecycle.",
        "treatment": [
            "Apply fungicide (myclobutanil or mancozeb) at pink bud stage through petal fall",
            "Remove nearby cedar/juniper hosts if feasible",
        ],
        "prevention": "Plant rust-resistant apple varieties. Remove galls from nearby cedars in winter.",
    },
    "Apple___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Blueberry
    "Blueberry___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Fungal disease causing white powdery coating on young leaves and shoots.",
        "treatment": [
            "Apply sulfur-based or potassium bicarbonate fungicide",
            "Improve air circulation through pruning",
            "Avoid excess nitrogen fertilization — lush growth is more susceptible",
        ],
        "prevention": "Plant resistant varieties. Avoid dense planting.",
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing rectangular gray-tan lesions that run parallel to leaf veins.",
        "treatment": [
            "Apply fungicide (strobilurin or triazole) at early sign",
            "Improve air circulation — avoid dense planting",
            "Till crop residue to reduce overwintering inoculum",
        ],
        "prevention": "Plant resistant hybrids. Practice crop rotation. Reduce surface residue.",
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Fungal disease causing small, oval, cinnamon-brown pustules on both leaf surfaces.",
        "treatment": [
            "Apply fungicide (propiconazole or azoxystrobin) if infection is severe",
            "Scout early — common rust is most damaging when it appears before tasseling",
        ],
        "prevention": "Plant rust-resistant hybrids. Early planting reduces exposure.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "High",
        "description": "Fungal disease causing large, cigar-shaped grayish-green lesions that can destroy the canopy.",
        "treatment": [
            "Apply fungicide (strobilurin + triazole mix) at the first sign of lesions",
            "Remove severely infected plant material",
        ],
        "prevention": "Use resistant hybrids. Rotate with non-host crops. Manage crop debris.",
    },
    "Corn_(maize)___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Grape
    "Grape___Black_rot": {
        "severity": "High",
        "description": "Fungal disease causing brown leaf lesions and hard, shriveled black mummified fruit.",
        "treatment": [
            "Apply fungicide (myclobutanil or mancozeb) from bud break through veraison",
            "Remove all mummified fruit — primary inoculum source",
            "Prune for good air circulation",
        ],
        "prevention": "Remove all mummies and debris before spring. Plant resistant varieties.",
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "High",
        "description": "Complex wood disease causing interveinal leaf scorch and internal wood decay. No cure.",
        "treatment": [
            "Remove and destroy infected wood — there is no effective cure",
            "Protect pruning wounds with fungicide paste or wound sealant",
            "Delay pruning to reduce infection risk",
        ],
        "prevention": "Avoid large pruning wounds. Use clean, sterilized tools.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "description": "Fungal disease causing angular dark lesions on older leaves, often with yellow halo.",
        "treatment": [
            "Apply copper-based or mancozeb fungicide",
            "Remove and destroy infected leaves",
            "Improve canopy air flow through leaf removal",
        ],
        "prevention": "Practice good canopy management. Remove leaf litter.",
    },
    "Grape___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Critical",
        "description": "Devastating bacterial disease spread by Asian citrus psyllid. No cure — infected trees must be removed.",
        "treatment": [
            "Remove and destroy infected trees immediately — there is no cure",
            "Control Asian citrus psyllid with systemic insecticide (imidacloprid)",
            "Report to local agricultural authority — this is a regulated pest",
        ],
        "prevention": "Use certified disease-free nursery stock. Install psyllid monitoring traps.",
    },

    # Peach
    "Peach___Bacterial_spot": {
        "severity": "High",
        "description": "Bacterial disease causing water-soaked leaf spots, shot holes, and fruit lesions.",
        "treatment": [
            "Apply copper-based bactericide at petal fall and every 10-14 days",
            "Avoid overhead irrigation",
            "Remove and destroy heavily infected shoots",
        ],
        "prevention": "Plant resistant varieties. Avoid sites with high wind — wounds increase infection.",
    },
    "Peach___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Raspberry
    "Raspberry___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Soybean
    "Soybean___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },

    # Squash
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Fungal disease causing white powdery coating on leaves, reducing photosynthesis.",
        "treatment": [
            "Apply potassium bicarbonate, neem oil, or sulfur fungicide",
            "Remove and destroy heavily infected leaves",
            "Avoid overhead watering — water at the base",
        ],
        "prevention": "Plant resistant varieties. Ensure good spacing for air circulation.",
    },

    # Strawberry
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "description": "Fungal disease causing small purple spots that enlarge and cause leaf margins to scorch.",
        "treatment": [
            "Apply fungicide (captan or myclobutanil)",
            "Remove infected leaves",
            "Avoid overhead irrigation",
        ],
        "prevention": "Use certified disease-free plants. Renovate beds regularly.",
    },
    "Strawberry___healthy": {
        "severity": "None",
        "description": "Plant appears healthy. No disease detected.",
        "treatment": ["No treatment needed"],
        "prevention": "Continue current practices. Monitor regularly.",
    },
}

SEVERITY_COLOR = {
    "None":     "green",
    "Moderate": "orange",
    "High":     "red",
    "Critical": "red",
}
