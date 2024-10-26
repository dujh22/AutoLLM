export const localPopupHandler = {
  key: 'local_popup_shown',
  get(): boolean {
    return window.localStorage.getItem(localPopupHandler.key) === '1';
  },
  set(local_popup_shown: boolean) {
    window.localStorage.setItem(localPopupHandler.key, local_popup_shown ? '1' : '0');
  },
};

export const localUserHandler = {
  key: 'local_user_data',
  get(): string {
    return window.localStorage.getItem(localUserHandler.key) || '';
  },
  set(phone: string) {
    window.localStorage.setItem(localUserHandler.key, phone);
  },
};
