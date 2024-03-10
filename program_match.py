class User:
    def __init__(self, category, race, religion, gender_identity, sexuality, location):
        self.category = category
        self.race = race
        self.religion = religion
        self.gender_identity = gender_identity
        self.sexuality = sexuality
        self.location = location


class Program:
    def __init__(self, name, category, race, religion, gender_identity, sexuality, service_delivery, location, link):
        self.name = name
        self.category = category
        self.race = race
        self.religion = religion
        self.gender_identity = gender_identity
        self.sexuality = sexuality
        self.service_delivery = service_delivery
        self.location = location
        self.link = link


def match_score(user, program):
    score = 0
    # Higher priority for location match
    if user.location == program.location:
        score += 5
    # Then check for religion match
    if user.religion == program.religion or program.religion == 15:  # '15' for not religious specific
        score += 4
    # Then health problem
    if user.category == program.category:
        score += 3
    # Then gender identity
    if user.gender_identity == program.gender_identity or program.gender_identity == 8:  # '8' for not gender identity specific
        score += 2
    # Lastly, sexuality
    if user.sexuality == program.sexuality or program.sexuality == 7:  # '7' for not sexuality specific
        score += 1
    return score


# Simulate user and programs input
user = User(category=1, race=9, religion=13, gender_identity=4, sexuality=3, location="ny")

programs = [
    Program("The Lesbian, Gay, Bisexual & Transgender Community Center", 1, 9, 15, 8, 7, 3, "ny", "https://gaycenter.org/"),
    Program("Spectrum Queer Community App", 3, 9, 15, 8, 7, 3, "ny", "https://spectrumapp.lgbt/"),
    Program("Liberate", 1, 7, 15, 8, 7, 1, "remote", "https://liberatemeditation.com/"),
    Program("The Caribbean Equality Project", 1, 8, 15, 8, 7, 3, "ny", "https://www.caribbeanequalityproject.org/mental-health-services"),
    Program("Callen-Lorde", 1, 9, 15, 7, 7, 2, "ny", "https://callen-lorde.org/"),
    Program("MASGD", 1, 9, 5, 8, 7, 1, "remote", "https://www.themasgd.org/"),
    Program("The Trevor Project", 2, 9, 15, 8, 7, 1, "remote", "https://www.thetrevorproject.org/resources/article/resources-for-sexual-health-support/"),
    Program("National Center For Lesbian Rights", 1, 9, 15, 2, 1, 3, "san francisco", "https://www.nclrights.org/"),
    Program("Rainbodhi", 1, 9, 8, 8, 7, 3, "international", "https://rainbodhi.org/"),
    Program("The Christian Closet", 1, 9, 1, 8, 7, 1, "remote", "https://www.thechristiancloset.com/"),
    Program("NQAPIA", 3, 2, 15, 8, 7, 2, "ny", "https://www.nqapia.org/"),
    Program("Connecticut State Department of Mental Health and Addiction Services", 1, 9, 15, 8, 7, 2, "Connecticut", "https://portal.ct.gov/DMHAS/Programs-and-services/Finding-Services/LGBT-Services"),
    Program("Inter Pride", 1, 9, 15, 8, 7, 1, "remote", "https://www.interpride.org/"),
    Program("Queer Hindu Alliance", 3, 2, 7, 8, 7, 3, "international", "https://www.clubhouse.com/house/lgbt-queer-hindu-alliance"),
    Program("Bisexual Resource Center", 2, 9, 15, 8, 3, 1, "remote", "https://biresource.org/"),
    Program("Ali Forney Center", 1, 9, 15, 8, 7, 2, "new york", "https://www.aliforneycenter.org/"),
    Program("National LGBTQIA+ Health Education Center", 2, 9, 15, 8, 7, 3, "boston", "https://www.lgbtqiahealtheducation.org/"),
    Program("Free Mom Hugs", 1, 9, 15, 8, 7, 2, "Oklahoma City", "https://freemomhugs.org/index.cfm?fuseaction=page.viewpage&pageid=4"),
    Program("The Okra Project", 1, 3, 15, 7, 7, 2, "ny", "https://www.theokraproject.com/"),
    Program("Marsha P. Johnson Institute", 1, 3, 15, 7, 7, 3, "national", "https://marshap.org/"),
    Program("Muslim for Progressive Values - Spiritual Counseling", 1, 2, 5, 8, 7, 1, "remote", "https://www.mpvusa.org/spiritual-counseling")
]

# Match user with programs
matches = [(program, match_score(user, program)) for program in programs]

# Sort matches based on score, higher first
matches.sort(key=lambda x: x[1], reverse=True)

i = 1
print("Highest 5 Programs for this user:\n")
for program, score in matches[:5]:
    print(f"Program {i}: {program.name}, Location: {program.location}, Link: {program.link}")
    i += 1
