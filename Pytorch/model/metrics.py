
def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den