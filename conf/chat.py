VOCAB = {
    "hint": {
        "minimal" : ["<recall>", "<critique>"],
        "standard": ["<recall>", "<critique>", "<diversify>"],
        "enhanced": ["<recall>", "<critique>", "<diversify>", "<reflect>"],
        "advanced": ["<recall>", "<critique>", "<diversify>", "<reflect>", "<plan>"]
    }
}

TEMPLATE = {
    "hint": {
        "messages": [
            {
                "role": "system",
                "content": {
                    "head": "Cognitive Operations Guide",
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
