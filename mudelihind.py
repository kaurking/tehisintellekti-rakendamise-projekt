
# 10 ainet = 6000 Ardi puhul. Mul rohkem ilmselt, plus * 2 (teine keel ka)
# valjund on umbes 500 selle puhul?

token_in = 6000 * 2
token_out = 500
messages = 10

def mudel(token_in, token_out, message_arv):
    mudel_in_price = 0.04
    mudel_out_price = 0.15

    return (token_in * mudel_in_price / 1_000_000 + token_out * mudel_out_price / 1_000_000) * message_arv

print(mudel(token_in, token_out, messages))