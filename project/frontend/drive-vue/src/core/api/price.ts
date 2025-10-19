import type { BodyPrice, Price } from '../types/price'
import { api } from './http'

export const fetchPrice = async (body: BodyPrice) => {
  const res = await api.post<Price[]>('/v1/calc/prices/', body)
  return res.data
}
