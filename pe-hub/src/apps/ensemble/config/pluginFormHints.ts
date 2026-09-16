/** Hover help text for the Add Model zip upload. */

export const PLUGIN_FORM_HINTS = {
  bundleZip:
    'Zip archive of the plugin directory: manifest.yaml, convert.py, wrapper.py, and optional weights/<id>/. Paths must not contain .. or absolute prefixes.',
  replaceExisting:
    'Overwrite a pending or rejected plugin with the same name. Active plugins must be deleted first.',
} as const
