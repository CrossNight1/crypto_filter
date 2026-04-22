import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatError(err: any): string {
  if (!err) return "";
  if (typeof err === 'string') return err;
  if (err.response?.data?.detail) {
    const detail = err.response.data.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d: any) => d.msg || JSON.stringify(d)).join(", ");
    return JSON.stringify(detail);
  }
  if (err.message) return err.message;
  return JSON.stringify(err);
}
