from src.extraction import CandidateProfile, anonymize, clean_text, extract_profile, make_candidate, read_cv

CV = """Name: Jane Doe
Gender: Female
Age: 34
Nationality: Canadian
jane.doe@mail.com | +33 6 12 34 56 78 | www.janedoe.dev

She worked as Data Scientist (2019 - 2023) where her models cut churn by 10%.
EDUCATION: Master's degree in Statistics
"""


def test_anonymize_removes_protected_attributes_but_keeps_dates():
    text, removed = anonymize(CV)
    low = text.lower()
    for leaked in ["jane", "female", "34", "canadian", "mail.com", "12 34 56", "she ", " her "]:
        assert leaked not in low, leaked
    assert "2019 - 2023" in text                 # date ranges are not phone numbers
    assert "data scientist" in low and "master" in low
    assert {"email", "phone", "url", "labelled_personal", "gender_words"} <= set(removed)


def test_read_txt_and_pdf(tmp_path):
    p = tmp_path / "cv.txt"
    p.write_text("Python developer", encoding="utf-8")
    assert read_cv(p) == "Python developer"

    from reportlab.pdfgen import canvas

    pdf = tmp_path / "cv.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Senior Java Developer 2018 - 2024")
    c.save()
    assert "Java Developer" in read_cv(pdf)
    assert "Java Developer" in read_cv(pdf.read_bytes(), "x.pdf")


def test_clean_text():
    assert clean_text("a  \t b\r\n\n\n\nc") == "a b\n\nc"


def test_make_candidate_and_extract(make_client, tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(CV, encoding="utf-8")
    cand = make_candidate(p)
    assert "Jane" not in cand.text and "Jane" in cand.raw_text
    client = make_client([{
        "skills": ["Python"], "years_experience": "4 years", "education": [
            {"degree": "Master's degree", "field": "Statistics", "level": "master"}],
        "languages": ["English"], "soft_skills": [], "certifications": [], "projects": [], "job_titles": []}])
    prof = extract_profile(cand.text, client)
    assert isinstance(prof, CandidateProfile)
    assert prof.years_experience == 4.0 and prof.education[0].level == "master"
    # the LLM only ever saw the anonymised text
    assert "Jane" not in client._backend.prompts[0]
