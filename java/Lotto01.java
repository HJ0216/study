import java.util.*;

public class Lotto01 {

	public static void main(String[] args) {
		int[] lotto = new int[6];
		
		Scanner scan = new Scanner(System.in);
		
		System.out.print("현금 입력: ");
		int input = scan.nextInt();
				
		for(int l=1; l<=input/1000; l++) {
//			input -= 1000;
//			if((l+1)%5==0) {System.out.println();}
			
			for(int i=0; i<lotto.length; i++) {
				lotto[i] = (int)(Math.random()*45+1);
								
				for(int j=0; j<i; j++) {
					if(lotto[i]==lotto[j]) {
						i--;
						break;}
					// continue, break 시, i가 증가된 채로 진행되므로 i를 감소시켜줘야 함
					// if - 참일 경우, 실행문이 2개 이상일 경우 { }로 묶어주기
				} // for inner
				
			} // for outer

		
		/*
		난수 제거
			  i			  j
		lotto[0]
		
		lotto[1] == lotto[0]
		
		lotto[2] == lotto[0]
		lotto[2] == lotto[1]

		lotto[3] == lotto[0]
		lotto[3] == lotto[1]
		lotto[3] == lotto[2]

		lotto[4] == lotto[0]
		lotto[4] == lotto[1]
		lotto[4] == lotto[2]
		lotto[4] == lotto[3]

		 */

			
			// Sort	
			for(int j=0; j<lotto.length-1; j++) {
				for(int k=j+1; k<lotto.length; k++) {
					if(lotto[j]>lotto[k]) {
						int tmp = lotto[k];
						lotto[k] = lotto[j];
						lotto[j] = tmp;
					}
				}
			}

			for(int lotto_data : lotto) {
				System.out.print(String.format("%5d", lotto_data));
			} System.out.println();

			if(l%5==0) {System.out.println();}	
			
		} // final
		
		scan.close();
		
	}
}

/*
[문제] 자동 Lotto
: 크기가 6개인 배열 생성
: 1-45 사이의 난수 발생(중복 제거)
: 오름차순 출력(Selection Sort 사용)
: 출력 시 자리 수 5자리
 */