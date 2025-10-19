export const useFormatTime = (timeInSeconds: number) => {
  return Math.round(timeInSeconds / 60)
}
