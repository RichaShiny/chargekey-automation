from charge_key_automation.app import ChargeKeyApp


def test_gui_callbacks_are_real_class_methods():
    assert callable(ChargeKeyApp._append)
    assert callable(ChargeKeyApp._failed)
    assert callable(ChargeKeyApp._open_output)
