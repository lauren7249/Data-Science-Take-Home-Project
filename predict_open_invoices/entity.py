class Entity:
    def __init__(self, id):
        self.id = id
        self.invoices = None

    @classmethod
    def get_open_invoices(self, entity_id=None, start_date_string=None, end_date_string=None):
        if entity_id is None:
            pass

    @classmethod
    def preprocess_invoices(self):
        pass