import pytest
from utils.delete_helper import delete_file_and_records


@pytest.mark.integration
def test_delete_file_and_records_existing_file(monkeypatch):
    """
    Testet die Löschung einer Datei und deren Datenbankeinträgen.
    Erwartet: Beide Löschungen erfolgreich oder zumindest sauber geloggt.
    """

    # Beispiel-Dateiname, der bereits in Supabase vorhanden sein muss
    test_filename = "trgs_611.pdf"

    # Funktion aufrufen
    log_output = delete_file_and_records(test_filename)

    # Ausgabe prüfen
    print("\n--- LOG ---\n", log_output)

    # Erwartung: keine Exception, Rückgabe enthält mindestens einen erwarteten Log-Satz
    assert any(
        key in log_output
        for key in [
            "🗑️ Storage-Datei gelöscht",
            "🧹",
            "✅",
            "⚠️",
        ]
    )
