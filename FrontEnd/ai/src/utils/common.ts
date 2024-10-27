import { IUpdateData } from '@/pages/home/types.ts';

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

export const localUpdateInfoHandler = {
  key: 'local_update_data',
  get(): IUpdateData | null {
    const res = window.localStorage.getItem(localUpdateInfoHandler.key);
    return res ? (JSON.parse(res) as IUpdateData) : null;
  },
  set(data: IUpdateData) {
    window.localStorage.setItem(localUpdateInfoHandler.key, JSON.stringify(data));
  },
};

export const localLastRefreshHandler = {
  key: 'local_last_refresh',
  get(): number {
    const res = window.localStorage.getItem(localLastRefreshHandler.key);
    return res ? +res : 0;
  },
  set(num: number) {
    window.localStorage.setItem(localLastRefreshHandler.key, num + '');
  },
};

export const getUpdateInfo = (): IUpdateData => {
  const lastRefresh = localLastRefreshHandler.get();
  const needRefresh = !lastRefresh || Date.now() - lastRefresh > 30 * 1000;

  const localData = localUpdateInfoHandler.get();

  const newNumBase = localData?.newNum || 25;
  const newNum = needRefresh ? newNumBase + Math.floor(Math.random() * 20) : newNumBase;

  const allNumBase = localData?.allNum || 501;
  const allNum = needRefresh ? allNumBase + Math.floor(Math.random() * 50) : allNumBase;

  const res = {
    newNum,
    allNum,
  };
  if (needRefresh) {
    localLastRefreshHandler.set(Date.now());
    localUpdateInfoHandler.set(res);
  }

  return res;
};

export const formatBriefContent = (str: string, links: string[]) => {
  const list = str.split('\n\n');
  // const title = `<span style="display: inline-block; font-weight: bold; font-size: 16px; margin-bottom: 8px">${list[0]}</span>`
  const linkList = list.slice(1).map((v, index) => `<a href="${links[index]}" target="_blank" style="display: inline-block; margin-bottom: 8px">${v}</a>`);
  return [...linkList].join('<br />').replace(/\*\*(.*?)\*\*：?/g, '<span style="display: inline-block; font-weight: bold; font-size: 16px;margin-bottom: 4px">$1</span><br />');
}

export const getNewLearn = () => {
  const targets = ['量子纠缠','机器学习', '共识算法', 'Word Embedding', '目标检测', '空间定位', 'IDS', '分布式存储']
  const max = 6, min = 3;
  const shuffled = targets.sort(() => 0.5 - Math.random());
  const count = Math.floor(Math.random() * (max - min + 1)) + min;
  return shuffled.slice(0, count).join(', ')
}
