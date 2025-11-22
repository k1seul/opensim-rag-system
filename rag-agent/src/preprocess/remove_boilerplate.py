import re

def remove_boilerplate(text):
    """Remove common UI/menu boilerplate from documentation text."""
    # Lowercase to match easily
    t = text.lower()

    # Common boilerplate patterns
    boilerplate_patterns = [
        r'atlassian uses cookies',
        r'accept all',
        r'on your device',
        r'atlassian and tracking notice',
        r'opens new window',
        r'only necessary',
        r'collapse sidebar',
        r'switch sites or apps',
        r'create create help log in',
        r'spaces apps opensim documentation',
        r'more actions back to top',
        r'shortcuts',
        r'this trigger is hidden',
        r'content results will update as you type',
        r'open[ -]?sim fellows',
        r'download',
        r'examples & tutorials',
        r'forum',
        r'upcoming events',
        r"user['’]?s guide",
    ]

    for pattern in boilerplate_patterns:
        t = re.sub(pattern, '', t)

    # Remove excessive whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t


# Usage example
if __name__ == '__main__':
    sample_text = """OpenSim+ Advanced Workshop March 2024 - OpenSim Documentation Atlassian uses cookies to improve your browsing experience. User's Guide Download Examples & Tutorials"""
    cleaned = remove_boilerplate(sample_text)
    print(cleaned)