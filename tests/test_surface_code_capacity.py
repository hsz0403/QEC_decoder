from surface_code_capacity import build_surface_code_iid_dataset


def test_surface_code_iid_shapes_and_parity():
    ds = build_surface_code_iid_dataset(distance=3, p=0.05, shots=32, seed=1, store_errors=True)
    assert ds["events"].shape[0] == 32
    assert ds["labels"].shape[0] == 32
    assert ds["parity_check"].shape[0] == ds["events"].shape[1]
    assert ds["logical_matrix"].shape[0] == ds["labels"].shape[1]
    errors = ds["errors"]
    recomputed_events = (errors @ ds["parity_check"].T) & 1
    recomputed_labels = (errors @ ds["logical_matrix"].T) & 1
    assert (recomputed_events == ds["events"]).all()
    assert (recomputed_labels == ds["labels"]).all()
