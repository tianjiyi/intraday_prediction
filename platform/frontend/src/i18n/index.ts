import { useUiStore } from '../stores/uiStore'
import en from './en'
import zh from './zh'

const dictionaries: Record<string, Record<string, string>> = { en, zh }

/** Returns a translation function bound to the current locale. */
export function useT(): (key: string) => string {
  const locale = useUiStore((s) => s.locale)
  return (key: string) => dictionaries[locale]?.[key] ?? dictionaries.en[key] ?? key
}
