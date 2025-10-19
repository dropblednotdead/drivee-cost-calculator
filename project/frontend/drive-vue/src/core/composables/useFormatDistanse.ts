export const useFormatDistance = (meters: number): number => {
  return Math.round((meters / 1000) * 10) / 10;
};
