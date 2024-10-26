import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { IChatItem } from '@/types';
import { localUserHandler } from '@/utils/common.ts';

export interface IStore {
  /* state */
  phone: string;
  chatList: IChatItem[];
  /* action */
  setPhone: (phone: string) => void;
  setChatList: (chatItem: IChatItem) => void;
}

export const useAiStore = create<IStore>()(
  devtools(
    (set, get) => ({
      phone: localUserHandler.get() || '',
      chatList: [],
      setPhone(phone: string) {
        set(() => ({ phone }));
        localUserHandler.set(phone);
      },
      setChatList(chatItem) {
        set(() => ({ chatList: [...get().chatList, chatItem] }));
      },
    }),
    { name: 'ai' }
  )
);
