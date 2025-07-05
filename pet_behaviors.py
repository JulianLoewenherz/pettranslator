# Pet Behavior Knowledge Base
# This provides context to the AI about what different pet behaviors typically mean

DOG_BEHAVIORS = {
    "body_language": {
        "tail_wagging": "Excitement, happiness, or arousal - speed and height indicate intensity",
        "play_bow": "Invitation to play, friendly social signal",
        "head_tilt": "Curiosity, trying to understand sounds or commands",
        "panting": "Normal cooling, excitement, or mild stress",
        "loose_body": "Relaxed, comfortable, friendly state",
        "stiff_body": "Alert, cautious, or potentially stressed",
        "rolling_over": "Submission, trust, or request for belly rubs",
        "jumping": "Excitement, greeting behavior, or attention-seeking"
    },
    "vocalizations": {
        "happy_barking": "Excited, playful, or greeting vocalizations",
        "alert_barking": "Warning about something in environment",
        "whining": "Excitement, anxiety, or wanting something",
        "growling": "Warning, discomfort, or playful (context dependent)",
        "howling": "Communication, response to sounds, or loneliness",
        "panting_sounds": "Normal breathing, excitement, or mild stress"
    },
    "facial_expressions": {
        "relaxed_eyes": "Comfortable, content, trusting",
        "wide_eyes": "Alert, excited, or potentially stressed",
        "ears_forward": "Alert, interested, paying attention",
        "ears_back": "Submissive, anxious, or conflicted",
        "mouth_open": "Relaxed, happy, or cooling down",
        "licking_lips": "Calming signal, mild stress, or anticipation"
    }
}

CAT_BEHAVIORS = {
    "body_language": {
        "tail_up": "Confident, friendly, happy to see you",
        "tail_swishing": "Agitated, overstimulated, or hunting mode",
        "slow_blinking": "Trust, contentment, 'cat kisses'",
        "kneading": "Comfort, happiness, reminiscent of nursing",
        "rolling": "Playful, comfortable, or attention-seeking",
        "arched_back": "Scared, defensive, or stretching",
        "crouching": "Hunting position, cautious, or fearful",
        "head_bumping": "Affection, marking with scent, social bonding"
    },
    "vocalizations": {
        "purring": "Contentment, comfort, or self-soothing",
        "meowing": "Communication with humans, requests or greetings",
        "chirping": "Excitement, hunting instincts, or bird-watching",
        "trilling": "Friendly greeting, mother-to-kitten communication",
        "hissing": "Fear, anger, or defensive warning",
        "chattering": "Excitement or frustration, often while watching prey"
    },
    "facial_expressions": {
        "half_closed_eyes": "Relaxed, trusting, content",
        "dilated_pupils": "Excited, scared, or overstimulated",
        "ears_forward": "Alert, interested, or curious",
        "ears_flattened": "Scared, angry, or defensive",
        "whiskers_forward": "Alert, interested, or hunting",
        "whiskers_back": "Defensive, scared, or cautious"
    }
}

def get_behavior_context(pet_type: str) -> str:
    """
    Generate behavior context string for AI prompt
    """
    if pet_type.lower() == "dog":
        behaviors = DOG_BEHAVIORS
        pet_name = "dog"
    elif pet_type.lower() == "cat":
        behaviors = CAT_BEHAVIORS
        pet_name = "cat"
    else:
        # Default to dog behaviors for unknown pets
        behaviors = DOG_BEHAVIORS
        pet_name = "pet"
    
    context = f"Common {pet_name} behaviors and their meanings:\n\n"
    
    for category, behavior_dict in behaviors.items():
        context += f"{category.replace('_', ' ').title()}:\n"
        for behavior, meaning in behavior_dict.items():
            context += f"  - {behavior.replace('_', ' ').title()}: {meaning}\n"
        context += "\n"
    
    return context

def get_pet_thoughts_examples(pet_type: str) -> list:
    """
    Get example thoughts for different pet types
    """
    if pet_type.lower() == "dog":
        return [
            "I'm so excited to see you! This is the best day ever!",
            "Something interesting is happening - I need to investigate!",
            "I want to play! Let's have some fun together!",
            "I'm feeling happy and relaxed right now.",
            "I'm trying to figure out what that sound was..."
        ]
    elif pet_type.lower() == "cat":
        return [
            "I'm feeling content and comfortable in this moment.",
            "Something has caught my attention - my hunting instincts are kicking in!",
            "I trust you completely and feel safe here.",
            "I'm in a playful mood and ready for some fun!",
            "I'm observing everything carefully from my comfortable spot."
        ]
    else:
        return [
            "I'm feeling happy and content right now!",
            "Something interesting is happening around me!",
            "I'm in a playful and energetic mood!",
            "I feel safe and comfortable in this environment.",
            "I'm curious about what's going on!"
        ]

# Common pet emotion mapping
EMOTION_KEYWORDS = {
    "happy": ["excited", "joyful", "content", "playful"],
    "curious": ["interested", "alert", "investigating", "wondering"],
    "relaxed": ["comfortable", "peaceful", "calm", "trusting"],
    "playful": ["energetic", "fun", "mischievous", "active"],
    "alert": ["watchful", "cautious", "attentive", "focused"]
} 