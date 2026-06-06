import os
import time
import requests

# ── GEMINI API KEY ──
GEMINI_API_KEY = "AIzaSyCQBh0nZm4tVAF1d2S9PoDfYs-eigOs3kU"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-lite-latest:generateContent?key=" + GEMINI_API_KEY
)

# ── FALLBACK LESSONS (used ONLY when API is unavailable) ──
FALLBACK_SLOW = (
    "Welcome to this step-by-step Pythagoras lesson designed for slow learners. "
    "A right triangle has one angle of exactly ninety degrees, called the right angle. "
    "The two shorter sides are called legs. The longest side, opposite the right angle, is the hypotenuse. "
    "Pythagoras discovered that if you square the first leg, square the second leg, and add them together, "
    "you get the square of the hypotenuse. We write this as a squared plus b squared equals c squared. "
    "For example, if one leg is three and the other is four: three times three is nine, four times four is sixteen. "
    "Nine plus sixteen equals twenty-five. The square root of twenty-five is five. "
    "So the hypotenuse is five. This works for every right triangle."
)


FALLBACK_ADVANCED = (
    "The Pythagorean theorem states c squared equals a squared plus b squared for right triangles. "
    "This extends to n-dimensional Euclidean space as the distance formula. "
    "For a equals three and b equals four, c equals the square root of twenty-five, which is five. "
    "The theorem underpins vector magnitude calculations, complex number modulus, and appears in physics energy equations. "
    "The converse also holds: if a squared plus b squared equals c squared, the triangle must be right-angled."
)


class GPTLessonGenerator:
    def __init__(self):
        self.topic = "Pythagoras theorem"
        self.max_retries = 2

    def generate_for_slow_learner(self):
        prompt = f"""You are an expert mathematics teacher creating an audio lesson for blind learners who are slow learners.
Generate a VERY DETAILED, STEP-BY-STEP audio script to teach the {self.topic}.

Requirements:
- Use VERY SIMPLE language, short sentences
- Break every concept into small steps
- Repeat key ideas for reinforcement
- Use the 3-4-5 triangle as a concrete example
- Explain WHY, not just HOW
- Total length: 400-500 words

Start directly without any preamble. Write for text-to-speech narration only — no markdown, no bullet points."""
        return self._call_gemini(prompt, FALLBACK_SLOW)


    def generate_for_advanced_learner(self):
        prompt = f"""You are an expert mathematics teacher creating an audio lesson for blind learners who are advanced learners.
Generate a concise, sophisticated audio script to teach the {self.topic}.

Requirements:
- Assume strong math foundation
- Focus on formula, proof concept, and real-world application
- Include a challenging extension or connection to higher math
- Total length: 200-300 words

Start directly without any preamble. Write for text-to-speech narration only — no markdown, no bullet points."""
        return self._call_gemini(prompt, FALLBACK_ADVANCED)

    def _call_gemini(self, prompt, fallback):
        """
        Call Google Gemini REST API with retries.
        Returns (text, source) where source is 'api' or 'fallback'.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"[Gemini] Attempt {attempt}/{self.max_retries} — calling Gemini API...")
                payload = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2048,
                        "thinkingConfig": {
                            "thinkingBudget": 0
                        }
                    }
                }
                response = requests.post(
                    GEMINI_URL,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Remove any markdown formatting for clean TTS
                text = text.replace("**", "").replace("*", "").replace("#", "").replace("\n\n", " ").replace("\n", " ")
                print(f"[Gemini] ✅ API call successful ({len(text)} chars)")
                return text, "api"

            except Exception as e:
                last_error = e
                print(f"[Gemini] ❌ Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(3)  # wait 3s before retry to avoid rate limit

        print(f"[Gemini] All attempts failed. Using fallback lesson.")
        print(f"[Gemini] Last error: {last_error}")
        return fallback, "fallback"

    def generate(self, level='Slow'):
        """Generate lesson based on learner level. Returns (text, source)."""
        level_lower = level.lower()
        if 'advanced' in level_lower:
            return self.generate_for_advanced_learner()
        else:
            # Slow learner (default) — also handles any average mapping
            return self.generate_for_slow_learner()
