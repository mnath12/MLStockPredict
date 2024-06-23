import { erf } from 'mathjs';

export function blackScholes(S, K, T, r, sigma, optionType = 'call') {
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    const N = x => 0.5 * (1 + erf(x / Math.sqrt(2)));
    
    if (optionType === 'call') {
        return S * N(d1) - K * Math.exp(-r * T) * N(d2);
    } else {
        return K * Math.exp(-r * T) * N(-d2) - S * N(-d1);
    }
}
