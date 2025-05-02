
def profile_model(model, inputs):
    from thop import profile
    from rich import print
    import sys
    macs, params = profile(model, inputs=inputs, verbose=False)
    print('모델 생성 완료! (Input {}: MACs: {} M | Params: {} K)'.format(
        sys.getsizeof(inputs[0].storage()),
        round(macs/1000/1000, 2), 
        round(params/1000, 2),
    ))