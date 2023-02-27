from tango.integrations.torch import Model


Model.register("dummy")
class DummyModel(Model):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args
        