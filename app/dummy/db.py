from .models.patient import Patient


class DummyDB:
    """
    Dummy class just as exemple on how to instanciate a class on app context
    """

    @staticmethod
    def patients() -> list[Patient]:
        """Create a dummy list of patients

        Returns:
        list[Patient]: dummy list of patients
        """

        patients = [
            Patient(1, 'Ricco', 'Le boss'),
            Patient(2, 'Francis', 'Cabrel'),
            Patient(3, 'Johnny', 'Hallyday'),
            Patient(4, 'Bob', 'Marley')
        ]

        return patients