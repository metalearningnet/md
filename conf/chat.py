VOCAB = {
    "hint": {
        "minimal" : ["<recall>", "<critique>"],
        "standard": ["<recall>", "<critique>", "<reflect>"],
        "enhanced": ["<recall>", "<critique>", "<reflect>", "<plan>"],
        "advanced": ["<recall>", "<critique>", "<reflect>", "<plan>", "<diversify>"]
    }
}

TEMPLATE = {
    "hint": {
        "messages": [
            {
                "role": "system",
                "content": {
                    "head": "Use clear line breaks between sentences to improve readability and organization. Cognitive Operations are as follows:",
                    "body": {
                        "<recall>": "Retrieve foundational knowledge (facts/rules/data patterns)",
                        "<critique>": "Evaluate solution quality (identify errors/assumptions/risks)",
                        "<diversify>": "Generate alternative solutions (explore novel approaches)",
                        "<reflect>": "Analyze process effectiveness (extract key learnings/improvements)",
                        "<plan>": "Construct action roadmap (define sequenced steps/resource allocation)"
                    }
                }
            }
        ]
    }
}
