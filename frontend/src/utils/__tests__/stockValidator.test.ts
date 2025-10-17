import { describe, expect, it } from 'vitest'

import {
  MarketType,
  validateMultipleStockCodes,
  validateStockCode,
} from '../stockValidator'


describe('validateStockCode', () => {
  it('accepts a well-formed Hong Kong stock code', () => {
    const result = validateStockCode('07000', MarketType.HK)

    expect(result.valid).toBe(true)
    expect(result.errorMessage).toBeUndefined()
  })

  it('rejects an invalid fund code with a helpful error message', () => {
    const result = validateStockCode('12AB56', MarketType.ETF)

    expect(result.valid).toBe(false)
    expect(result.errorMessage).toContain('基金代码应为6位数字')
  })
})

describe('validateMultipleStockCodes', () => {
  it('collects all invalid codes for the selected market', () => {
    const invalid = validateMultipleStockCodes(
      ['07000', 'ABCDE', '123'],
      MarketType.HK,
    )

    expect(invalid).toHaveLength(2)
    expect(invalid.map((entry) => entry.code)).toEqual(['ABCDE', '123'])
  })
})
