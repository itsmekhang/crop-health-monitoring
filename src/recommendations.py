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
}

SEVERITY_COLOR = {
    "None":     "green",
    "Moderate": "orange",
    "High":     "red",
    "Critical": "red",
}
