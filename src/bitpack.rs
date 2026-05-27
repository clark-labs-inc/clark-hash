#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackedCodes {
    bits: u8,
    len: usize,
    bytes: Vec<u8>,
}

impl PackedCodes {
    pub(crate) fn new(bits: u8, len: usize) -> Self {
        let total_bits = len.saturating_mul(bits as usize);
        let total_bytes = total_bits.div_ceil(8);
        Self {
            bits,
            len,
            bytes: vec![0_u8; total_bytes],
        }
    }

    pub(crate) fn bits(&self) -> u8 {
        self.bits
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn set(&mut self, index: usize, value: u8) {
        debug_assert!(index < self.len);
        let mask = (1_u16 << self.bits) - 1;
        let value = (value as u16) & mask;
        let bit_offset = index * self.bits as usize;
        let byte_offset = bit_offset / 8;
        let intra = bit_offset % 8;

        let lo = self.bytes[byte_offset] as u16;
        let hi = self.bytes.get(byte_offset + 1).copied().unwrap_or(0_u8) as u16;
        let mut chunk = lo | (hi << 8);

        chunk &= !(mask << intra);
        chunk |= value << intra;

        self.bytes[byte_offset] = chunk as u8;
        if byte_offset + 1 < self.bytes.len() {
            self.bytes[byte_offset + 1] = (chunk >> 8) as u8;
        }
    }

    pub(crate) fn get(&self, index: usize) -> u8 {
        debug_assert!(index < self.len);
        let mask = (1_u16 << self.bits) - 1;
        let bit_offset = index * self.bits as usize;
        let byte_offset = bit_offset / 8;
        let intra = bit_offset % 8;

        let lo = self.bytes[byte_offset] as u16;
        let hi = self.bytes.get(byte_offset + 1).copied().unwrap_or(0_u8) as u16;
        let chunk = lo | (hi << 8);

        ((chunk >> intra) & mask) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::PackedCodes;

    #[test]
    fn roundtrip_for_all_supported_bit_widths() {
        for bits in 1_u8..=8 {
            let len = 37;
            let mut packed = PackedCodes::new(bits, len);
            let max_value = ((1_u16 << bits) - 1) as u8;

            for idx in 0..len {
                packed.set(idx, (idx as u8) & max_value);
            }

            for idx in 0..len {
                assert_eq!(packed.get(idx), (idx as u8) & max_value);
            }
        }
    }
}
