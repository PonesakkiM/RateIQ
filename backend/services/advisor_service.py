"""
RateIQ – Natural Language AI Advisor Service
Acts as an AI Product Manager — interprets NL queries and generates
data-driven responses using ML predictions, SHAP values, and rule-based reasoning.
"""
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger("rateiq.advisor")

# ── Intent taxonomy ────────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "low_rating":       [r"why.*low.*rating", r"rating.*low", r"bad.*rating", r"poor.*rating", r"low.*star"],
    "improve_rating":   [r"improve.*rating", r"boost.*rating", r"increase.*rating", r"better.*rating", r"higher.*rating", r"how.*raise"],
    "improve_installs": [r"improve.*install", r"more.*install", r"increase.*download", r"boost.*download", r"get.*more.*user"],
    "size_question":    [r"reduce.*size", r"app.*size", r"size.*too.*big", r"large.*app", r"compress"],
    "price_question":   [r"should.*free", r"price.*strategy", r"paid.*free", r"freemium", r"charge.*app", r"reduce.*price", r"pricing"],
    "ads_question":     [r"remove.*ads", r"ads.*impact", r"ads.*rating", r"should.*ads", r"monetize"],
    "update_question":  [r"update.*frequen", r"how.*often.*update", r"when.*update", r"stale.*app", r"last.*update"],
    "reviews_question": [r"more.*review", r"get.*review", r"review.*strateg", r"ask.*review", r"prompt.*review"],
    "competitor_question": [r"competitor", r"competition", r"similar.*app", r"compare.*app", r"market.*position"],
    "category_question": [r"which.*categor", r"change.*categor", r"best.*categor", r"categor.*strategy", r"wrong.*categor"],
    "screenshots":      [r"screenshot", r"store.*image", r"play.*store.*visual", r"listing.*image"],
    "general_advice":   [r"what.*do", r"help.*me", r"advice", r"suggest", r"recommend", r"tips", r"how.*improve"],
}

# ── Rule-based reasoning engine ───────────────────────────────────────────────
SHAP_LABEL_MAP = {
    "log_reviews":        "Review Count",
    "update_days":        "Days Since Update",
    "log_installs":       "Install Count",
    "has_ads":            "Contains Ads",
    "is_free":            "Free App",
    "size_mb":            "App Size",
    "price":              "Price",
    "num_screenshots":    "Screenshots Count",
    "category_enc":       "App Category",
    "content_rating_enc": "Content Rating",
}

FEATURE_THRESHOLDS = {
    "update_days":     {"bad": 180,  "warn": 90,  "good": 30},
    "num_screenshots": {"bad": 2,    "warn": 3,   "good": 5},
    "reviews":         {"bad": 100,  "warn": 500, "good": 5_000},
    "size_mb":         {"bad": 150,  "warn": 80,  "good": 30},
}


def _detect_intent(query: str) -> List[str]:
    """Detect one or more intents from a natural language query."""
    query_lower = query.lower()
    detected = []
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, query_lower):
                detected.append(intent)
                break
    if not detected:
        detected = ["general_advice"]
    return detected


def _extract_issues(app_data: dict, shap_values: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Analyze app data and SHAP values to find root-cause issues.
    Returns list of issues with severity, description, and fix suggestions.
    """
    issues = []
    rating = app_data.get("predicted_rating", 3.8)

    # Check each feature
    update_days = app_data.get("update_days", 30)
    if update_days > FEATURE_THRESHOLDS["update_days"]["bad"]:
        issues.append({
            "severity": "critical",
            "feature": "Last Update",
            "issue": f"App not updated in {update_days} days (>{FEATURE_THRESHOLDS['update_days']['bad']} days is critical)",
            "root_cause": "Stale apps lose user trust and algorithm favor. Play Store penalizes apps with low update frequency.",
            "fix": f"Ship an update within the next 2 weeks. Even minor bug fixes count. Target updates every 30-60 days.",
            "expected_impact": "+0.2 to +0.4 ★",
        })
    elif update_days > FEATURE_THRESHOLDS["update_days"]["warn"]:
        issues.append({
            "severity": "warning",
            "feature": "Last Update",
            "issue": f"App last updated {update_days} days ago — approaching stale threshold",
            "root_cause": "Infrequent updates reduce perceived quality and user engagement.",
            "fix": "Plan a maintenance release within 30 days to stay fresh.",
            "expected_impact": "+0.1 to +0.2 ★",
        })

    num_screenshots = app_data.get("num_screenshots", 3)
    if num_screenshots < FEATURE_THRESHOLDS["num_screenshots"]["bad"]:
        issues.append({
            "severity": "critical",
            "feature": "Store Screenshots",
            "issue": f"Only {num_screenshots} screenshot(s) — critically low visual presence",
            "root_cause": "Screenshots are a key conversion driver. Low count signals low effort and hurts store conversion rate.",
            "fix": "Add 5-8 high-quality screenshots showcasing your app's best features. Use captions.",
            "expected_impact": "+0.1 to +0.3 ★",
        })
    elif num_screenshots < FEATURE_THRESHOLDS["num_screenshots"]["good"]:
        issues.append({
            "severity": "warning",
            "feature": "Store Screenshots",
            "issue": f"Only {num_screenshots} screenshots — below the recommended 5-8",
            "root_cause": "Fewer screenshots reduce listing quality and user trust.",
            "fix": "Add screenshots highlighting key features, UI, and value proposition.",
            "expected_impact": "+0.05 to +0.1 ★",
        })

    reviews = app_data.get("reviews", 1_000)
    if reviews < FEATURE_THRESHOLDS["reviews"]["bad"]:
        issues.append({
            "severity": "critical",
            "feature": "Review Count",
            "issue": f"Only {reviews} reviews — insufficient social proof",
            "root_cause": "Low review count means users can't gauge app quality from others' experiences. Trust barrier is high.",
            "fix": "Implement smart in-app review prompts after successful task completion. Respond to existing reviews.",
            "expected_impact": "+0.15 to +0.4 ★ (compounding)",
        })
    elif reviews < FEATURE_THRESHOLDS["reviews"]["warn"]:
        issues.append({
            "severity": "warning",
            "feature": "Review Count",
            "issue": f"{reviews} reviews — below 500 threshold for strong social proof",
            "root_cause": "Apps with fewer than 500 reviews are perceived as less established.",
            "fix": "Prompt users to review at high-satisfaction moments (e.g., after completing a goal).",
            "expected_impact": "+0.1 to +0.2 ★",
        })

    has_ads = app_data.get("has_ads", 0)
    if has_ads and rating < 4.2:
        issues.append({
            "severity": "warning",
            "feature": "In-App Ads",
            "issue": "App contains ads — contributing to below-average rating",
            "root_cause": "Ads create friction and frustration, especially in low-rated apps. They reduce session quality.",
            "fix": "Reduce ad frequency or introduce an ad-free paid tier. Use rewarded ads instead of interstitials.",
            "expected_impact": "+0.1 to +0.2 ★",
        })

    size_mb = app_data.get("size_mb", 30)
    if size_mb > FEATURE_THRESHOLDS["size_mb"]["bad"]:
        issues.append({
            "severity": "warning",
            "feature": "App Size",
            "issue": f"App size is {size_mb:.1f} MB — heavy apps have higher uninstall rates",
            "root_cause": "Large apps are skipped during installs and uninstalled first when storage is low, reducing review count.",
            "fix": "Optimize assets, use on-demand delivery for large modules, compress images.",
            "expected_impact": "Indirect: improved install rate → more reviews → higher rating",
        })

    price = app_data.get("price", 0.0)
    if price > 0 and rating < 4.0:
        issues.append({
            "severity": "warning",
            "feature": "Pricing",
            "issue": f"Paid app (${price:.2f}) with predicted rating below 4.0 — value perception gap",
            "root_cause": "Users expect paid apps to meet higher quality standards. Sub-4.0 ratings on paid apps deter purchases.",
            "fix": "Improve core quality, or offer a free trial period before purchase.",
            "expected_impact": "Indirect: better quality perception → higher purchase-to-review ratio",
        })

    # SHAP-based issues (if available)
    if shap_values:
        for shap in shap_values[:3]:
            if shap.get("value", 0) < -0.05:
                feat = shap.get("feature", "")
                label = shap.get("label", feat)
                if feat not in [i.get("shap_feature", "") for i in issues]:
                    issues.append({
                        "severity": "info",
                        "feature": label,
                        "issue": f"{label} is your {abs(shap['value']):.3f}-star drag on predicted rating",
                        "root_cause": f"ML model identified {label} as a significant negative driver.",
                        "fix": "See feature-specific recommendations in the Prediction Engine tab.",
                        "expected_impact": f"Up to +{abs(shap['value']):.2f} ★",
                        "shap_feature": feat,
                    })

    return issues


def _generate_response(intents: List[str], app_data: dict, issues: List[Dict], prediction_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Build structured AI response based on detected intents and analysis."""
    
    rating = app_data.get("predicted_rating", prediction_data.get("prediction", 3.8) if prediction_data else 3.8)
    category = app_data.get("category", "your category")

    primary_intent = intents[0]
    response_text = ""
    recommendations = []
    follow_up_questions = []

    if primary_intent == "low_rating":
        critical = [i for i in issues if i["severity"] == "critical"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        
        response_text = f"**Why your rating is predicted at {rating:.1f} ★:**\n\n"
        if critical:
            response_text += f"I found **{len(critical)} critical issue(s)** that are significantly dragging your rating down:\n\n"
            for i, iss in enumerate(critical[:3], 1):
                response_text += f"{i}. **{iss['feature']}**: {iss['root_cause']}\n"
        elif warnings:
            response_text += f"No critical issues, but **{len(warnings)} warning(s)** are limiting your rating ceiling:\n\n"
            for i, iss in enumerate(warnings[:3], 1):
                response_text += f"{i}. **{iss['feature']}**: {iss['issue']}\n"
        else:
            response_text += f"Your {rating:.1f} rating is primarily driven by:\n- Natural market variation in {category}\n- Room for polish and user engagement improvements.\n"

        recommendations = [{"title": i["feature"], "action": i["fix"], "impact": i.get("expected_impact", ""), "severity": i["severity"]} for i in issues[:4]]
        follow_up_questions = ["How can I improve my review count?", "Should I reduce my app size?", "What is my competitor gap?"]

    elif primary_intent == "improve_rating":
        response_text = f"**Roadmap to improve your {rating:.1f} ★ rating:**\n\n"
        response_text += "As your AI Product Manager, here's a prioritized action plan:\n\n"
        
        priority_actions = []
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        warning_issues  = [i for i in issues if i["severity"] == "warning"]
        
        if critical_issues:
            response_text += "**🔴 Priority 1 — Fix Critical Issues:**\n"
            for iss in critical_issues[:2]:
                response_text += f"- {iss['fix']} ({iss.get('expected_impact', '')})\n"
                priority_actions.append({"title": f"Fix: {iss['feature']}", "action": iss["fix"], "impact": iss.get("expected_impact", ""), "severity": "critical"})
        
        if warning_issues:
            response_text += "\n**🟡 Priority 2 — Address Warnings:**\n"
            for iss in warning_issues[:2]:
                response_text += f"- {iss['fix']} ({iss.get('expected_impact', '')})\n"
                priority_actions.append({"title": f"Improve: {iss['feature']}", "action": iss["fix"], "impact": iss.get("expected_impact", ""), "severity": "warning"})
        
        response_text += f"\n**📈 Potential:**  With these fixes, your predicted rating could reach **{min(5.0, rating + 0.5):.1f}–{min(5.0, rating + 0.8):.1f} ★**."
        recommendations = priority_actions or [{"title": "Maintain quality", "action": "Keep update cadence and respond to all user reviews", "impact": "Sustains rating", "severity": "info"}]
        follow_up_questions = ["Why is my rating low?", "How do I get more reviews?", "What do competitors do differently?"]

    elif primary_intent == "improve_installs":
        response_text = "**How to increase your app's install count:**\n\n"
        response_text += "Installs are a function of **visibility + conversion rate**. Focus on:\n\n"
        response_text += "1. **ASO (App Store Optimization)**: Keyword-rich title, description, and tags\n"
        response_text += "2. **Visual assets**: High-quality icon + 6-8 compelling screenshots → improves conversion\n"
        response_text += "3. **Rating quality**: Apps above 4.2 ★ get 30-40% more organic installs via algorithmic boost\n"
        response_text += "4. **App size**: Each 10 MB of extra size reduces install conversion by ~5% on limited data plans\n"
        response_text += "5. **Pricing**: Free apps get 10-100x more installs than equivalents priced at $0.99+\n"
        recommendations = [
            {"title": "Optimize Store Listing", "action": "Use relevant keywords in title and description", "impact": "+15-30% organic discovery", "severity": "info"},
            {"title": "Improve Visual Assets", "action": "Add 6-8 professional screenshots with feature highlights", "impact": "+10-20% conversion rate", "severity": "info"},
            {"title": "Reduce App Size", "action": f"Target under 50 MB (current: {app_data.get('size_mb', '?')} MB)", "impact": "+5-15% install rate", "severity": "warning" if app_data.get("size_mb", 30) > 50 else "info"},
        ]
        follow_up_questions = ["How do screenshots affect installs?", "Should I make my app free?"]

    elif primary_intent == "size_question":
        size_mb = app_data.get("size_mb", 30)
        response_text = f"**App size analysis: {size_mb:.1f} MB**\n\n"
        if size_mb > 100:
            response_text += f"⚠️ At {size_mb:.1f} MB, your app is significantly larger than the category average.\n\n"
            response_text += "**Why it matters:**\n- 25% of users skip apps >100 MB on mobile data\n- Large apps are first to be uninstalled under storage pressure\n- Higher uninstall rate → fewer long-term reviews\n\n"
            response_text += "**How to reduce size:**\n1. Use WebP/AVIF images instead of PNG/JPEG\n2. Enable on-demand resource delivery (Android App Bundles)\n3. Remove unused libraries and assets\n4. Compress audio/video assets"
        elif size_mb > 50:
            response_text += f"Your app at {size_mb:.1f} MB is above average. Consider optimization but it's not critical.\n"
        else:
            response_text += f"Your app size of {size_mb:.1f} MB is reasonable and shouldn't negatively impact ratings or installs."
        recommendations = [{"title": "Reduce APK size", "action": "Use Android App Bundle (AAB) for up to 30% size reduction", "impact": "Improved install conversion", "severity": "warning" if size_mb > 80 else "info"}]
        follow_up_questions = ["How does app size affect installs?", "What is Android App Bundle?"]

    elif primary_intent == "price_question":
        price = app_data.get("price", 0.0)
        is_free = app_data.get("is_free", 1)
        response_text = "**Pricing strategy analysis:**\n\n"
        if is_free:
            response_text += "Your free app model is aligned with market expectations.\n\n"
            response_text += "**If considering monetization:**\n- Freemium (free core + paid features) performs best for sustained growth\n- Subscriptions outperform one-time purchases for long-term revenue\n- Avoid excessive ads — they cost you ~0.15 ★ on average"
        else:
            response_text += f"Your app is priced at **${price:.2f}**.\n\n"
            response_text += "**Market context:** 80%+ of top-rated apps are free. Paid apps face a much higher quality bar.\n\n"
            response_text += "**Recommendations:**\n1. Consider a freemium model with a generous free tier\n2. Add a 7-day free trial if staying paid\n3. Ensure your rating is at least 4.2 before keeping the paid model — users expect more for paid apps"
        recommendations = [{"title": "Pricing model review", "action": "A/B test freemium vs. paid with a small user segment", "impact": "Potentially 10x more installs", "severity": "info"}]
        follow_up_questions = ["What is the freemium model?", "How do ads affect my rating?"]

    elif primary_intent == "ads_question":
        has_ads = app_data.get("has_ads", 0)
        response_text = "**Ad monetization analysis:**\n\n"
        if has_ads:
            response_text += f"Your app has ads enabled. In the dataset, apps with ads average **0.13-0.18 ★ lower** than equivalents without ads.\n\n"
            response_text += "**Options:**\n1. **Remove all ads** → Recover ~0.15 ★ but need alternative revenue\n2. **Rewarded ads only** → User-initiated, much lower negative impact\n3. **Ad-free tier** → Offer paid version without ads, captures quality-sensitive users\n4. **Reduce ad frequency** → Drop interstitials, keep banners only"
        else:
            response_text += "Your app currently has no ads. This is a positive signal for ratings.\n\n"
            response_text += "If you plan to add monetization, consider:\n- Non-intrusive banner ads (lowest rating impact)\n- Rewarded video ads (users choose to view)\n- Subscription model (highest long-term value, minimal rating impact)"
        recommendations = [{"title": "Optimize ad strategy", "action": "Replace interstitials with rewarded ads" if has_ads else "Avoid intrusive ad formats", "impact": "+0.1 to +0.2 ★" if has_ads else "Maintain clean rating", "severity": "warning" if has_ads else "info"}]
        follow_up_questions = ["What ad formats are least disruptive?", "How do I switch to a subscription model?"]

    elif primary_intent == "update_question":
        update_days = app_data.get("update_days", 30)
        response_text = "**Update frequency analysis:**\n\n"
        if update_days > 180:
            response_text += f"⚠️ Your app hasn't been updated in **{update_days} days** — this is a critical signal.\n\n"
            response_text += "**Impact:**\n- Users see stale 'Last Updated' date and lose confidence\n- Play Store algorithm may deprioritize stale apps\n- Bug reports accumulate and unaddressed issues compound\n\n"
            response_text += "**Action plan:**\n1. Ship a maintenance release ASAP (even minor fixes count)\n2. Establish a 4-6 week release cadence\n3. Use automated CI/CD to reduce release friction"
        elif update_days > 90:
            response_text += f"Your app was last updated **{update_days} days ago** — approaching the stale threshold.\n\nPlan a release within the next 3-4 weeks."
        else:
            response_text += f"Your update frequency looks good (last update: {update_days} days ago). Keep this cadence."
        recommendations = [{"title": "Establish release cadence", "action": "Ship updates every 30-45 days, even minor improvements", "impact": "+0.1 to +0.3 ★", "severity": "critical" if update_days > 180 else "info"}]
        follow_up_questions = ["What should I include in updates?", "How do I automate releases?"]

    elif primary_intent == "reviews_question":
        reviews = app_data.get("reviews", 1_000)
        response_text = f"**Review strategy: Current count = {reviews:,} reviews**\n\n"
        response_text += "Reviews are the #1 factor in user trust and Play Store ranking.\n\n"
        response_text += "**Proven tactics to increase reviews:**\n"
        response_text += "1. **In-app review prompts**: Use Google's `ReviewManager` API after user achieves a goal\n"
        response_text += "2. **Timing matters**: Prompt after completing a level, finishing a task, or receiving positive feedback — NOT on first launch\n"
        response_text += "3. **Respond to all reviews**: Responding shows you care — boosts future review rates by 15-25%\n"
        response_text += "4. **Fix negative reviews**: Acknowledge issues, fix them, reply with fix details — often converts 1-star to 3-4 star\n"
        response_text += "5. **Community building**: Discord/Reddit community members become vocal reviewers"
        recommendations = [{"title": "Implement review prompts", "action": "Add Google Play In-App Review API after positive moments", "impact": "+50-200% more reviews monthly", "severity": "info"}]
        follow_up_questions = ["How do I respond to negative reviews?", "When is the best time to ask for reviews?"]

    elif primary_intent == "competitor_question":
        response_text = f"**Competitive positioning in {category}:**\n\n"
        response_text += "Head to the **Competitor Gap Analyzer** tab for a full side-by-side analysis.\n\n"
        response_text += f"**Quick take:** In {category}, top apps typically:\n"
        response_text += "- Have 3-5x more reviews than average apps\n"
        response_text += "- Update every 30-45 days\n"
        response_text += "- Offer a free tier (even if paid features exist)\n"
        response_text += "- Have 6-8 optimized store screenshots"
        recommendations = [{"title": "Run Competitor Analysis", "action": "Use the Competitor Gap Analyzer tab for detailed comparison", "impact": "Identify specific gaps", "severity": "info"}]
        follow_up_questions = ["Who are my top competitors?", "What are competitor rating strategies?"]

    elif primary_intent == "screenshots":
        screenshots = app_data.get("num_screenshots", 3)
        response_text = f"**Store screenshot analysis: {screenshots} screenshots**\n\n"
        response_text += "Screenshots are your silent salesperson on the Play Store listing.\n\n"
        response_text += "**Best practices:**\n"
        response_text += "1. **Quantity**: 6-8 screenshots is the sweet spot\n"
        response_text += "2. **First 2 screenshots**: Most users see only these — make them show your #1 value\n"
        response_text += "3. **Add text captions**: Feature highlights in 3-5 words per screenshot\n"
        response_text += "4. **Use device frames**: Professional mockups on phone images convert 20% better\n"
        response_text += "5. **Show real UI**: Avoid marketing stock photos — show what the app actually does\n"
        if screenshots < 5:
            response_text += f"\n⚠️ You have only {screenshots} screenshot(s). This is below optimal and hurts conversion rate."
        recommendations = [{"title": "Improve store screenshots", "action": f"Add {max(0, 6 - screenshots)} more screenshots with feature captions", "impact": "+5-15% install conversion", "severity": "warning" if screenshots < 4 else "info"}]
        follow_up_questions = ["What tools can I use to create screenshots?", "Should I add a promo video?"]

    else:  # general_advice
        response_text = f"**AI Product Manager Summary for your app:**\n\n"
        response_text += f"Based on your predicted rating of **{rating:.1f} ★**, here's your overall health:\n\n"
        
        if issues:
            critical_count = len([i for i in issues if i["severity"] == "critical"])
            warning_count  = len([i for i in issues if i["severity"] == "warning"])
            response_text += f"- 🔴 Critical issues: {critical_count}\n- 🟡 Warnings: {warning_count}\n\n"
        
        response_text += "**Top 3 actions to take right now:**\n"
        top_issues = sorted(issues, key=lambda x: 0 if x["severity"] == "critical" else 1)[:3]
        for i, iss in enumerate(top_issues, 1):
            response_text += f"{i}. {iss['fix']}\n"
        
        if not issues:
            response_text += "1. Maintain regular update cadence (every 30-45 days)\n"
            response_text += "2. Actively engage with user reviews\n"
            response_text += "3. A/B test store listing visuals for better conversion\n"
        
        recommendations = [{"title": i["feature"], "action": i["fix"], "impact": i.get("expected_impact", ""), "severity": i["severity"]} for i in top_issues]
        follow_up_questions = ["Why is my rating low?", "How can I improve installs?", "What do competitors do better?"]

    return {
        "response": response_text,
        "detected_intents": intents,
        "issues_found": len(issues),
        "issue_summary": issues[:5],
        "recommendations": recommendations,
        "follow_up_questions": follow_up_questions,
        "confidence": "high" if len(issues) > 0 else "medium",
    }


def process_chat(
    query: str,
    app_data: dict,
    prediction_data: Optional[Dict] = None,
    chat_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Main chat processing function.
    
    Args:
        query: User's natural language question
        app_data: Current app metadata (from prediction form or session)
        prediction_data: Last prediction result (optional, for SHAP context)
        chat_history: Previous conversation turns (optional)
    
    Returns:
        Structured response with text, recommendations, and follow-ups
    """
    if not query or not query.strip():
        return {
            "response": "Please ask me something about your app — e.g., 'Why is my rating low?' or 'How can I improve installs?'",
            "detected_intents": [],
            "issues_found": 0,
            "issue_summary": [],
            "recommendations": [],
            "follow_up_questions": ["Why is my rating low?", "How can I improve installs?", "Should I reduce app size?"],
            "confidence": "low",
        }

    # Detect intent
    intents = _detect_intent(query)
    logger.info("Chat query: '%s' → intents: %s", query[:80], intents)

    # Enrich app_data with prediction if available
    enriched_data = dict(app_data)
    shap_values = None
    if prediction_data:
        enriched_data["predicted_rating"] = prediction_data.get("prediction", 3.8)
        shap_values = prediction_data.get("shap_values", [])

    # Extract issues
    issues = _extract_issues(enriched_data, shap_values)

    # Generate response
    response = _generate_response(intents, enriched_data, issues, prediction_data)

    # Add context-aware greeting
    if chat_history and len(chat_history) > 0:
        context_note = ""
    else:
        context_note = f"*I'm your AI Product Manager. I have access to your app data and can analyze {enriched_data.get('category', 'your')} category trends.*\n\n"
        response["response"] = context_note + response["response"]

    return response
